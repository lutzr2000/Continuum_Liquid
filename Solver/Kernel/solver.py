import numpy as np
from pathlib import Path

from . import output
from . import pressure_solve
from . import boundary_conditions
from . import particles as particle_ops
from . import velocity_update


def solver(output_path):
    t_start = 0.0
    t_end = 10.0
    fps = 24
    dt = 1.0 / fps

    Lx = 5
    Ly = 5
    Lz = 5
    delta = 0.1

    rho_air = 1.225
    rho_water = 1000.0
    nu_air = 1.5e-5
    nu_water = 1.0e-6

    nx = int(Lx / delta)
    ny = int(Ly / delta)
    nz = int(Lz / delta)
    dx = delta

    jacobi_iterations = 100
    gravity = np.array([0.0, -9.81, 0.0])
    fx = 0.0
    fy = 0.0
    fz = 0.0

    x = np.arange(1, nx - 1)[:, None, None]

    # Linke 10 % der Domain mit Wasser füllen
    initial_fluid = np.broadcast_to(
        x < 0.2 * nx,
        (nx - 2, ny - 2, nz - 2)
    )

    cells = np.argwhere(initial_fluid) + 1

    particles_per_axis = 2  # 2³ = 8 particles per cell

    offsets_1d = (np.arange(particles_per_axis) + 0.5) / particles_per_axis

    offsets = np.stack(
        np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    particles = (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 3) * dx

    particle_velocity = np.zeros((len(particles), 3), dtype=np.float64)

    output_dir = Path(output_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "water_particles.usdc"

    total_frames = round((t_end - t_start) * fps)

    stage, usd_points = output.create_usd(
        filepath=output_file, fps=fps, start_frame=1, end_frame=total_frames + 1
    )

    output.write_particles(usd_points, particles, frame=1)

    for step in range(total_frames):
        t = t_start + step * dt

        print(
            f"Frame {step + 1}/{total_frames} | "
            f"Time: {t:.3f}s | "
            f"Particles: {len(particles)}"
        )

        velocity, fluid, cell = particle_ops.particle_to_grid(
            particles, particle_velocity, (nx, ny, nz), dx
        )

        velocity = boundary_conditions.grid_domain_bc(velocity)

        old_velocity = velocity.copy()

        velocity = velocity_update.update_velocity(
            velocity, fluid, gravity, fx, fy, fz, rho_water, nu_water, dx, dt
        )

        pressure = pressure_solve.pressure_solve(
            velocity, fluid, dx, dt, rho_water, jacobi_iterations
        )

        velocity = pressure_solve.pressure_correction(
            velocity, pressure, dx, dt, rho_water
        )

        particle_velocity = particle_ops.grid_to_particle(
            particle_velocity, velocity, old_velocity, cell
        )

        particles = particle_ops.advect_particles(particles, particle_velocity, dt)

        particles, particle_velocity = boundary_conditions.particle_domain_bc(
            particles, particle_velocity, (nx, ny, nz), dx
        )

        output.write_particles(usd_points, particles, frame=step + 2)

    output.save_usd(stage)
