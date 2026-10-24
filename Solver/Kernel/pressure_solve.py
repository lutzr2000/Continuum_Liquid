import numpy as np

from . import boundary_conditions
from . import helper
from . import kernel_config
from . import multigrid

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def pressure_solve(
    kernels,
    multigrid_kernels,
    boundary_kernels,
    queue,
    velocity,
    fluid_fraction,
    pressure,
    rhs,
    p_levels,
    b_levels,
    mask_levels,
    operator_level0,
    operator_levels,
    multigrid_level_shapes,
    dx,
    dt,
    rho,
    num_vcycles,
    grid_shape,
):
    nx, ny, nz = grid_shape
    grid_cell_count = nx * ny * nz
    u, v, w = velocity

    helper.fill_device(queue, pressure, 0.0)
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
    multigrid_kernels["build_finest_operator"](
        queue,
        (grid_cell_count,),
        None,
        fluid_fraction,
        *operator_level0,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )

    for _ in range(num_vcycles):
        multigrid.v_cycle(
            multigrid_kernels,
            queue,
            0,
            p_levels,
            b_levels,
            mask_levels,
            operator_levels,
            pressure,
            rhs,
            fluid_fraction,
            operator_level0,
            dx,
            multigrid_level_shapes,
            grid_shape,
            pre_smooth=2,
            post_smooth=4,
            coarse_smooth_iterations=20,
        )

    boundary_conditions.neumann_domain_bc(boundary_kernels, queue, pressure, grid_shape)
    return pressure


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
