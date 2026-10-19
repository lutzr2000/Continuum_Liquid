from pathlib import Path

import numpy as np
import pyopencl as cl

from . import boundary_conditions
from . import helper
from . import kernel_config
from . import output
from . import particles as particle_ops
from . import pressure_solve
from . import velocity_update

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def solver(config):
    simulation_config = config["simulation"]
    domain_config = simulation_config["domain"]
    domain_size = domain_config["size"]
    physics_config = simulation_config["physics"]
    water_config = physics_config["water"]
    air_config = physics_config["air"]

    t_start = 0.0
    t_end = 10.0
    fps = 24
    dt = 1.0 / fps

    Lx = float(domain_size["x"])
    Ly = float(domain_size["y"])
    Lz = float(domain_size["z"])
    delta = float(domain_config["resolution"])

    rho_water = float(water_config["density"])
    nu_water = float(water_config["viscosity"])
    rho_air = float(air_config["density"])
    nu_air = float(air_config["viscosity"])

    nx = int(Lx / delta)
    ny = int(Ly / delta)
    nz = int(Lz / delta)
    grid_shape = (nx, ny, nz)
    grid_cell_count = nx * ny * nz
    dx = delta

    jacobi_iterations = 50
    gravity = np.array([0.0, -9.81, 0.0], dtype=FIELD_DTYPE)
    fx = 0.0
    fy = 0.0
    fz = 0.0

    x = np.arange(1, nx - 1)[:, None, None]
    initial_fluid = np.broadcast_to(x < 0.2 * nx, (nx - 2, ny - 2, nz - 2))
    cells = np.argwhere(initial_fluid) + 1

    particles_per_axis = 2
    particles_per_cell = particles_per_axis**3
    offsets_1d = (np.arange(particles_per_axis) + 0.5) / particles_per_axis
    offsets = np.stack(
        np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    particles_host = np.ascontiguousarray(
        (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 3) * dx,
        dtype=FIELD_DTYPE,
    )
    particle_velocity_host = np.zeros_like(particles_host)
    number_of_particles = len(particles_host)

    device = helper.select_opencl_device()
    kernel_config.configure_for_device(device, "CPU")
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device=device)

    kernel_path = Path(__file__).parent / "OpenCL"
    particles_kernels = helper.load_program(context, kernel_path / "particles.cl")
    velocity_update_kernels = helper.load_program(
        context, kernel_path / "velocity_update.cl"
    )
    pressure_solve_kernels = helper.load_program(
        context, kernel_path / "pressure_solve.cl"
    )
    boundary_conditions_kernels = helper.load_program(
        context, kernel_path / "boundary_conditions.cl"
    )
    kernel_sets = (
        particles_kernels,
        velocity_update_kernels,
        pressure_solve_kernels,
        boundary_conditions_kernels,
    )
    kernel_config.constrain_to_compiled_kernels(
        (kernel for kernel_set in kernel_sets for kernel in kernel_set.values()),
        device,
    )

    particles = helper.to_device(context, particles_host)
    particle_velocity = helper.to_device(context, particle_velocity_host)

    velocity = tuple(helper.device_array(context, grid_shape) for _ in range(3))
    velocity_work = tuple(helper.device_array(context, grid_shape) for _ in range(3))
    old_velocity = tuple(helper.device_array(context, grid_shape) for _ in range(3))
    weight = helper.device_array(context, grid_shape)
    particle_count = helper.device_array(context, grid_shape, dtype=np.int32)
    fluid_fraction = helper.device_array(context, grid_shape)
    pressure = helper.device_array(context, grid_shape)
    pressure_work = helper.device_array(context, grid_shape)
    pressure_rhs = helper.device_array(context, grid_shape)
    pressure_diagonal = helper.device_array(context, grid_shape)

    boundary_conditions.create_pressure_boundary(
        boundary_conditions_kernels,
        queue,
        pressure_diagonal,
        grid_shape,
    )

    output_dir = Path(simulation_config["output_path"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "water_particles.usdc"
    total_frames = round((t_end - t_start) * fps)
    stage, usd_points = output.create_usd(
        filepath=output_file, fps=fps, start_frame=1, end_frame=total_frames + 1
    )
    output.write_particles(usd_points, particles_host, frame=1)

    for step in range(total_frames):
        t = t_start + step * dt
        print(
            f"Frame {step + 1}/{total_frames} | "
            f"Time: {t:.3f}s | "
            f"Particles: {number_of_particles}"
        )

        velocity, fluid_fraction = particle_ops.particle_to_grid(
            particles_kernels,
            queue,
            particles,
            particle_velocity,
            velocity,
            weight,
            particle_count,
            fluid_fraction,
            number_of_particles,
            grid_shape,
            dx,
            particles_per_cell,
        )
        velocity = boundary_conditions.grid_domain_bc(
            boundary_conditions_kernels, queue, velocity, grid_shape
        )

        for old_component, component in zip(old_velocity, velocity):
            cl.enqueue_copy(queue, old_component, component)

        velocity_work = velocity_update.update_velocity(
            velocity_update_kernels,
            queue,
            velocity,
            velocity_work,
            fluid_fraction,
            gravity,
            fx,
            fy,
            fz,
            rho_water,
            nu_water,
            dx,
            dt,
            grid_shape,
        )
        velocity, velocity_work = velocity_work, velocity

        pressure, pressure_work = pressure_solve.pressure_solve(
            pressure_solve_kernels,
            boundary_conditions_kernels,
            queue,
            velocity,
            fluid_fraction,
            pressure,
            pressure_work,
            pressure_rhs,
            pressure_diagonal,
            dx,
            dt,
            rho_water,
            jacobi_iterations,
            grid_shape,
        )
        velocity = pressure_solve.pressure_correction(
            pressure_solve_kernels,
            queue,
            velocity,
            pressure,
            dx,
            dt,
            rho_water,
            grid_shape,
        )

        particle_velocity = particle_ops.grid_to_particle(
            particles_kernels,
            queue,
            particle_velocity,
            particles,
            velocity,
            old_velocity,
            number_of_particles,
            grid_shape,
            dx,
        )
        particles = particle_ops.advect_particles(
            particles_kernels,
            queue,
            particles,
            particle_velocity,
            number_of_particles,
            dt,
        )
        particles, particle_velocity = boundary_conditions.particle_domain_bc(
            boundary_conditions_kernels,
            queue,
            particles,
            particle_velocity,
            number_of_particles,
            grid_shape,
            dx,
        )

        cl.enqueue_copy(queue, particles_host, particles).wait()
        output.write_particles(usd_points, particles_host, frame=step + 2)

    output.save_usd(stage)
