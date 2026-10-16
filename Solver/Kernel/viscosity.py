import numpy as np


def apply_viscosity(velocity, nu, dx, dt):

    if nu <= 0.0:
        return velocity

    laplacian = (
        velocity[2:, 1:-1, 1:-1, :]
        + velocity[:-2, 1:-1, 1:-1, :]
        + velocity[1:-1, 2:, 1:-1, :]
        + velocity[1:-1, :-2, 1:-1, :]
        + velocity[1:-1, 1:-1, 2:, :]
        + velocity[1:-1, 1:-1, :-2, :]
        - 6.0 * velocity[1:-1, 1:-1, 1:-1, :]
    ) / (dx * dx)

    result = velocity.copy()

    result[1:-1, 1:-1, 1:-1, :] += (
        nu * dt * laplacian
    )

    return result