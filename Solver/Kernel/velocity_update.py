import numpy as np


def update_velocity(velocity, fluid, gravity, fx, fy, fz, rho, nu, dx, dt):
    force = np.array([fx, fy, fz], dtype=velocity.dtype)
    result = velocity.copy()
    result[fluid] += (gravity + force / rho) * dt

    laplacian = (
        velocity[2:, 1:-1, 1:-1, :]
        + velocity[:-2, 1:-1, 1:-1, :]
        + velocity[1:-1, 2:, 1:-1, :]
        + velocity[1:-1, :-2, 1:-1, :]
        + velocity[1:-1, 1:-1, 2:, :]
        + velocity[1:-1, 1:-1, :-2, :]
        - 6.0 * velocity[1:-1, 1:-1, 1:-1, :]
    ) / (dx * dx)

    result[1:-1, 1:-1, 1:-1, :] += nu * dt * laplacian

    return result
