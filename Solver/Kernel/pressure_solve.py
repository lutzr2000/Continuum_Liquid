import numpy as np
from . import boundary_conditions

def pressure_solve(velocity, fluid, dx, dt, rho, jacobi_iterations):
    divergence = np.zeros(fluid.shape, dtype=np.float64)
    divergence[1:-1, 1:-1, 1:-1] = (
        velocity[2:, 1:-1, 1:-1, 0]
        - velocity[:-2, 1:-1, 1:-1, 0]
        + velocity[1:-1, 2:, 1:-1, 1]
        - velocity[1:-1, :-2, 1:-1, 1]
        + velocity[1:-1, 1:-1, 2:, 2]
        - velocity[1:-1, 1:-1, :-2, 2]
    ) / (2.0 * dx)

    pressure = np.zeros(fluid.shape, dtype=np.float64)
    rhs = rho / dt * divergence

    diagonal = boundary_conditions.create_pressure_boundary(fluid.shape)

    for _ in range(jacobi_iterations):
        pressure_new = np.zeros_like(pressure)
        neighbor_sum = (
            pressure[2:, 1:-1, 1:-1] * fluid[2:, 1:-1, 1:-1]
            + pressure[:-2, 1:-1, 1:-1] * fluid[:-2, 1:-1, 1:-1]
            + pressure[1:-1, 2:, 1:-1] * fluid[1:-1, 2:, 1:-1]
            + pressure[1:-1, :-2, 1:-1] * fluid[1:-1, :-2, 1:-1]
            + pressure[1:-1, 1:-1, 2:] * fluid[1:-1, 1:-1, 2:]
            + pressure[1:-1, 1:-1, :-2] * fluid[1:-1, 1:-1, :-2]
        )
        interior_diagonal = diagonal[1:-1, 1:-1, 1:-1]
        pressure_new[1:-1, 1:-1, 1:-1] = np.divide(
            neighbor_sum - rhs[1:-1, 1:-1, 1:-1] * dx * dx,
            interior_diagonal,
            out=np.zeros_like(neighbor_sum),
            where=interior_diagonal > 0.0,
        )
        pressure = np.where(fluid, pressure_new, 0.0)

    return pressure


def pressure_correction(velocity, pressure, dx, dt, rho):
    # Ghostdruck an festen Wänden: p_ghost = p_fluid, also dp/dn = 0.
    pressure_with_walls = boundary_conditions.apply_pressure_boundary_conditions(
        pressure
    )

    velocity[1:-1, 1:-1, 1:-1, 0] -= (
        dt
        / rho
        * (
            pressure_with_walls[2:, 1:-1, 1:-1]
            - pressure_with_walls[:-2, 1:-1, 1:-1]
        )
        / (2.0 * dx)
    )
    velocity[1:-1, 1:-1, 1:-1, 1] -= (
        dt
        / rho
        * (
            pressure_with_walls[1:-1, 2:, 1:-1]
            - pressure_with_walls[1:-1, :-2, 1:-1]
        )
        / (2.0 * dx)
    )
    velocity[1:-1, 1:-1, 1:-1, 2] -= (
        dt
        / rho
        * (
            pressure_with_walls[1:-1, 1:-1, 2:]
            - pressure_with_walls[1:-1, 1:-1, :-2]
        )
        / (2.0 * dx)
    )

    return velocity
