import numpy as np


def _trilinear_stencil(particles, grid_shape, dx):
    grid_position = particles / dx - 0.5
    base = np.floor(grid_position).astype(np.int32)
    base = np.clip(base, 0, np.asarray(grid_shape) - 2)
    fraction = np.clip(grid_position - base, 0.0, 1.0)

    stencil = []
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                offset = np.array([ox, oy, oz], dtype=np.int32)
                weight = np.prod(
                    np.where(offset == 1, fraction, 1.0 - fraction), axis=1
                )
                stencil.append((base + offset, weight))

    return stencil


def particle_to_grid(
    particles, particle_velocity, grid_shape, dx, particles_per_cell
):
    nx, ny, nz = grid_shape
    cell = np.clip(
        (particles / dx).astype(np.int32),
        1,
        [nx - 2, ny - 2, nz - 2],
    )

    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    weight = np.zeros((nx, ny, nz), dtype=np.float32)

    stencil = _trilinear_stencil(particles, grid_shape, dx)
    for indices, interpolation_weight in stencil:
        index = (indices[:, 0], indices[:, 1], indices[:, 2])
        np.add.at(weight, index, interpolation_weight)
        for component in range(3):
            np.add.at(
                velocity[..., component],
                index,
                interpolation_weight * particle_velocity[:, component],
            )

    valid_velocity = weight > 0.0
    velocity[valid_velocity] /= weight[valid_velocity, None]

    particle_count = np.zeros((nx, ny, nz), dtype=np.int32)
    np.add.at(particle_count, (cell[:, 0], cell[:, 1], cell[:, 2]), 1)
    particle_volume = dx**3 / particles_per_cell
    liquid_volume = particle_count * particle_volume
    fluid_fraction = liquid_volume / dx**3

    return velocity, fluid_fraction, stencil


def grid_to_particle(particle_velocity, velocity, old_velocity, stencil):
    grid_change = velocity - old_velocity
    particle_change = np.zeros_like(particle_velocity)

    for indices, interpolation_weight in stencil:
        particle_change += (
            interpolation_weight[:, None]
            * grid_change[indices[:, 0], indices[:, 1], indices[:, 2]]
        )

    particle_velocity += particle_change

    return particle_velocity


def advect_particles(particles, particle_velocity, dt):
    particles += particle_velocity * dt

    return particles
