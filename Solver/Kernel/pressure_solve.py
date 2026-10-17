import numpy as np

from . import boundary_conditions
from . import helper
from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def pressure_solve(
    kernels,
    boundary_kernels,
    queue,
    velocity,
    fluid_fraction,
    pressure,
    pressure_work,
    rhs,
    diagonal,
    dx,
    dt,
    rho,
    jacobi_iterations,
    grid_shape,
):
    nx, ny, nz = grid_shape
    grid_cell_count = nx * ny * nz
    u, v, w = velocity

    helper.fill_device(queue, pressure, 0.0)
    helper.fill_device(queue, pressure_work, 0.0)
    kernels["pressure_rhs"](
        queue,
        (grid_cell_count,),
        None,
        u,
        v,
        w,
        fluid_fraction,
        rhs,
        FIELD_DTYPE(rho / dt),
        FIELD_DTYPE(1.0 / dx),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )

    for _ in range(jacobi_iterations):
        kernels["pressure_solve"](
            queue,
            (grid_cell_count,),
            None,
            pressure,
            pressure_work,
            rhs,
            diagonal,
            fluid_fraction,
            FIELD_DTYPE(dx * dx),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        pressure, pressure_work = pressure_work, pressure

    boundary_conditions.neumann_domain_bc(boundary_kernels, queue, pressure, grid_shape)
    return pressure, pressure_work


def pressure_correction(
    kernels,
    queue,
    velocity,
    pressure,
    dx,
    dt,
    rho,
    grid_shape,
):
    nx, ny, nz = grid_shape
    u, v, w = velocity
    kernels["pressure_correction"](
        queue,
        (nx * ny * nz,),
        None,
        u,
        v,
        w,
        pressure,
        FIELD_DTYPE(dt / rho),
        FIELD_DTYPE(1.0 / dx),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    return velocity
