import numpy as np

def particle_to_grid(particles, particle_velocity, grid_shape, dx):
    nx, ny, nz = grid_shape
    cell = np.clip(
        (particles / dx).astype(np.int32),
        1,
        [nx - 2, ny - 2, nz - 2],
    )

    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    weight = np.zeros((nx, ny, nz), dtype=np.float32)

    for component in range(3):
        np.add.at(
            velocity[..., component],
            (cell[:, 0], cell[:, 1], cell[:, 2]),
            particle_velocity[:, component],
        )

    np.add.at(weight, (cell[:, 0], cell[:, 1], cell[:, 2]), 1.0)

    fluid = weight > 0.0
    velocity[fluid] /= weight[fluid, None]

    return velocity, fluid, cell


def grid_to_particle(particle_velocity, velocity, old_velocity, cell):
    grid_change = velocity - old_velocity
    particle_velocity += grid_change[cell[:, 0], cell[:, 1], cell[:, 2]]

    return particle_velocity


def advect_particles(particles, particle_velocity, dt):
    particles += particle_velocity * dt

    return particles
