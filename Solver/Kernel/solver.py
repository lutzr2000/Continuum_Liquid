from pathlib import Path

import numpy as np
import pyopencl as cl

from ..General.main import emit_message
from . import boundary_conditions
from . import helper
from . import kernel_config
from . import multigrid
from . import output
from . import particles as particle_methods
from . import pressure_solve
from . import timing
from . import velocity_update

FIELD_DTYPE = kernel_config.FIELD_DTYPE


@timing.profiled_run
def solver(config: dict, timings=None):
    simulation = config.get("simulation") or {}
    settings = simulation.get("settings") or {}

    # ------------device-------------------
    device = helper.select_opencl_device()
    kernel_config.configure_for_device(
        device,
        "CPU",
    )

    # ------------context-------------------
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device=device)

    kernel_path = Path(__file__).parent / "OpenCL"

    particles_kernels = helper.load_program(
        context,
        kernel_path / "particles.cl",
    )

    velocity_update_kernels = helper.load_program(
        context,
        kernel_path / "velocity_update.cl",
    )

    pressure_solve_kernels = helper.load_program(
        context,
        kernel_path / "pressure_solve.cl",
    )

    multigrid_kernels = helper.load_program(
        context,
        kernel_path / "multigrid.cl",
    )

    boundary_conditions_kernels = helper.load_program(
        context,
        kernel_path / "boundary_conditions.cl",
    )

    kernel_sets = (
        particles_kernels,
        velocity_update_kernels,
        pressure_solve_kernels,
        multigrid_kernels,
        boundary_conditions_kernels,
    )
    kernel_config.constrain_to_compiled_kernels(
        (kernel for kernel_set in kernel_sets for kernel in kernel_set.values()),
        device,
    )

    print("################################################################")

    # ------------config-------------------
    domain = simulation.get("domain") or {}
    domain_size = domain.get("size") or {}
    physics = simulation.get("physics") or {}
    water = physics.get("water") or {}
    air = physics.get("air") or {}
    output_dir = Path(simulation["output_path"]).expanduser().resolve()
    cancel_event = (config.get("meta") or {}).get("cancel_event")

    # ------------time-------------------
    start_frame = int(settings.get("start_frame", 1))
    end_frame = int(settings.get("end_frame", 250))

    fps = int(settings.get("fps", 24))
    substeps = max(0, int(settings.get("substeps", 1)))
    steps_per_frame = substeps + 1

    total_frames = max(0, end_frame - start_frame)
    total_solver_steps = total_frames * steps_per_frame

    dt = 1.0 / (fps * steps_per_frame)

    # ------------dimensions------------------
    Lx = float(domain_size["x"])
    Ly = float(domain_size["y"])
    Lz = float(domain_size["z"])
    delta = float(domain["resolution"])

    nx = int(Lx / delta)
    ny = int(Ly / delta)
    nz = int(Lz / delta)
    grid_shape = (nx, ny, nz)

    # ------------physics------------------
    rho_water = float(water["density"])
    nu_water = float(water["viscosity"])
    rho_air = float(air["density"])
    nu_air = float(air["viscosity"])

    multigrid_cycles = max(1, int(settings.get("iterations", 1)))
    gravity = np.array([0.0, 0.0, -9.81], dtype=FIELD_DTYPE)
    fx = 0.0
    fy = 0.0
    fz = 0.0

    # ------------particles------------------
    x = np.arange(1, nx - 1)[:, None, None]
    initial_fluid = np.broadcast_to(x < 0.5 * nx, (nx - 2, ny - 2, nz - 2))
    cells = np.argwhere(initial_fluid) + 1

    particles_per_axis = 2
    particles_per_cell = particles_per_axis**3
    offsets_1d = (np.arange(particles_per_axis) + 0.5) / particles_per_axis
    offsets = np.stack(
        np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)

    particles_host = np.ascontiguousarray(
        (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 3) * delta,
        dtype=FIELD_DTYPE,
    )
    particle_velocity_host = np.zeros_like(particles_host)
    number_of_particles = len(particles_host)
    print(f"Total particle count: {number_of_particles}")

    # ------------device arrays------------------
    # partciles
    particles = helper.to_device(context, particles_host)
    particle_velocity = helper.to_device(context, particle_velocity_host)
    (
        particle_work,
        particle_velocity_work,
        particle_bin_counts,
        particle_bin_scan_outputs,
        particle_bin_scan_sums,
    ) = particle_methods.create_particle_sort_buffers(
        context,
        grid_shape,
        number_of_particles,
    )
    particle_count = helper.device_array(context, grid_shape, dtype=np.int32)

    # velocity
    u = helper.device_array(context, grid_shape)
    v = helper.device_array(context, grid_shape)
    w = helper.device_array(context, grid_shape)

    u_work = helper.device_array(context, grid_shape)
    v_work = helper.device_array(context, grid_shape)
    w_work = helper.device_array(context, grid_shape)

    # pressure
    pressure = helper.device_array(context, grid_shape)
    pressure_rhs = helper.device_array(context, grid_shape)

    # ------------multigrid------------------
    (
        multigrid_pressure_levels,
        multigrid_rhs_levels,
        multigrid_mask_levels,
        multigrid_delta_levels,
        multigrid_level_shapes,
    ) = multigrid.create_multigrid_levels(
        context,
        grid_shape,
        delta,
        min_size=8,
    )

    # flip
    fluid_fraction = helper.device_array(context, grid_shape)

    # ------------output------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    particle_cache = output.create_cache(
        directory=output_dir,
        fps=fps,
        start_frame=start_frame,
        end_frame=end_frame,
        particle_count=number_of_particles,
    )
    output.write_particles(particle_cache, particles_host, frame=start_frame)
    emit_message({"type": "stats", "frame": 0, "total_frames": total_frames})

    # ------------time loop------------------
    t = 0
    for solver_step in range(total_solver_steps):
        if cancel_event is not None and cancel_event.is_set():
            print("Bake cancellation requested. Stopping the simulation cleanly...")
            break

        t += solver_step * dt
        output_frame = (solver_step + 1) // steps_per_frame
        substep_index = solver_step % steps_per_frame

        # ------------particle to grid-------------------
        with timings.section("solver", "particle_to_grid", queue):
            (
                particles,
                particle_velocity,
                particle_work,
                particle_velocity_work,
                (u, v, w),
                fluid_fraction,
            ) = particle_methods.particle_to_grid(
                particles_kernels,
                queue,
                particles,
                particle_velocity,
                particle_work,
                particle_velocity_work,
                (u, v, w),
                particle_bin_counts,
                particle_bin_scan_outputs,
                particle_bin_scan_sums,
                particle_count,
                fluid_fraction,
                number_of_particles,
                grid_shape,
                delta,
                particles_per_cell,
            )

        # ------------grid bc-------------------
        with timings.section("solver", "grid_domain_bc", queue):
            u, v, w = boundary_conditions.grid_domain_bc(
                boundary_conditions_kernels,
                queue,
                (u, v, w),
                grid_shape,
            )

        # ------------velocity update-------------------
        with timings.section("solver", "update_velocity", queue):
            u_work, v_work, w_work = velocity_update.update_velocity(
                velocity_update_kernels,
                queue,
                (u, v, w),
                (u_work, v_work, w_work),
                fluid_fraction,
                gravity,
                fx,
                fy,
                fz,
                rho_water,
                nu_water,
                delta,
                dt,
                grid_shape,
            )

        # ------------swap-------------------
        u, u_work = u_work, u
        v, v_work = v_work, v
        w, w_work = w_work, w

        # Forces and viscosity are evaluated on every interior entry, which
        # includes the positive-side MAC wall faces.  Close them again before
        # constructing the pressure RHS.
        with timings.section("solver", "updated_grid_domain_bc", queue):
            u, v, w = boundary_conditions.grid_domain_bc(
                boundary_conditions_kernels,
                queue,
                (u, v, w),
                grid_shape,
            )

        # ------------pressure solve-------------------
        with timings.section("solver", "pressure_solve", queue):
            pressure = pressure_solve.pressure_solve(
                pressure_solve_kernels,
                multigrid_kernels,
                boundary_conditions_kernels,
                queue,
                (u, v, w),
                fluid_fraction,
                pressure,
                pressure_rhs,
                multigrid_pressure_levels,
                multigrid_rhs_levels,
                multigrid_mask_levels,
                multigrid_delta_levels,
                multigrid_level_shapes,
                delta,
                dt,
                rho_water,
                multigrid_cycles,
                grid_shape,
            )

        # ------------pressure correction-------------------
        with timings.section("solver", "pressure_correction", queue):
            u, v, w = pressure_solve.pressure_correction(
                pressure_solve_kernels,
                queue,
                (u, v, w),
                pressure,
                delta,
                dt,
                rho_water,
                grid_shape,
            )

        # Projection changes the face velocities, including solid-wall faces.
        # Re-apply the MAC boundary condition before transferring back to the
        # particles so no normal velocity can survive at the domain wall.
        with timings.section("solver", "projected_grid_domain_bc", queue):
            u, v, w = boundary_conditions.grid_domain_bc(
                boundary_conditions_kernels,
                queue,
                (u, v, w),
                grid_shape,
            )

        # ------------grid to particle-------------------
        with timings.section("solver", "grid_to_particle", queue):
            particle_velocity = particle_methods.grid_to_particle(
                particles_kernels,
                queue,
                particle_velocity,
                particles,
                (u, v, w),
                (u_work, v_work, w_work),
                number_of_particles,
                grid_shape,
                delta,
            )

        # ------------advect particles-------------------
        with timings.section("solver", "advect_particles", queue):
            particles = particle_methods.advect_particles(
                particles_kernels,
                queue,
                particles,
                particle_velocity,
                number_of_particles,
                dt,
            )

        # ------------particle boundary conditions-------------------
        with timings.section("solver", "particle_domain_bc", queue):
            particles, particle_velocity = boundary_conditions.particle_domain_bc(
                boundary_conditions_kernels,
                queue,
                particles,
                particle_velocity,
                number_of_particles,
                grid_shape,
                delta,
            )

        # ------------output-------------------
        if substep_index == steps_per_frame - 1:
            with timings.section("transfer", "particles_to_host", queue):
                cl.enqueue_copy(queue, particles_host, particles).wait()

            with timings.section("output", "write_particles"):
                output.write_particles(
                    particle_cache,
                    particles_host,
                    frame=start_frame + output_frame,
                )
            emit_message(
                {
                    "type": "stats",
                    "frame": output_frame,
                    "total_frames": total_frames,
                }
            )
