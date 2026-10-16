import numpy as np


def create_pressure_boundary(grid_shape):
    solid = np.zeros(grid_shape, dtype=bool)
    solid[[0, -1], :, :] = True
    solid[:, [0, -1], :] = True
    solid[:, :, [0, -1]] = True

    diagonal = np.zeros(grid_shape, dtype=np.float64)
    diagonal[1:-1, 1:-1, 1:-1] = (
        (~solid[2:, 1:-1, 1:-1]).astype(np.float64)
        + (~solid[:-2, 1:-1, 1:-1]).astype(np.float64)
        + (~solid[1:-1, 2:, 1:-1]).astype(np.float64)
        + (~solid[1:-1, :-2, 1:-1]).astype(np.float64)
        + (~solid[1:-1, 1:-1, 2:]).astype(np.float64)
        + (~solid[1:-1, 1:-1, :-2]).astype(np.float64)
    )

    return diagonal


def apply_pressure_boundary_conditions(pressure):
    pressure_with_walls = pressure.copy()
    pressure_with_walls[0, :, :] = pressure_with_walls[1, :, :]
    pressure_with_walls[-1, :, :] = pressure_with_walls[-2, :, :]
    pressure_with_walls[:, 0, :] = pressure_with_walls[:, 1, :]
    pressure_with_walls[:, -1, :] = pressure_with_walls[:, -2, :]
    pressure_with_walls[:, :, 0] = pressure_with_walls[:, :, 1]
    pressure_with_walls[:, :, -1] = pressure_with_walls[:, :, -2]

    return pressure_with_walls


def apply_boundary_conditions(velocity):
    velocity[[0, -1], :, :, :] = 0.0
    velocity[:, [0, -1], :, :] = 0.0
    velocity[:, :, [0, -1], :] = 0.0

    return velocity

def apply_particle_boundary(particles, particle_velocity, grid_shape, dx):
    lower = np.full(3, 1.001 * dx)
    upper = (np.asarray(grid_shape) - 1.001) * dx
    hit_wall = (particles < lower) | (particles > upper)

    particles = np.clip(particles, lower, upper)
    particle_velocity[hit_wall] = 0.0

    return particles, particle_velocity
