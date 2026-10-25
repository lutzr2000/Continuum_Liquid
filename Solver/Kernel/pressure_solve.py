import numpy as np
import pyopencl as cl

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
    divergence_work,
    divergence_host,
    fluid_fraction_host,
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

    cl.enqueue_copy(queue, fluid_fraction_host, fluid_fraction).wait()
    active_fluid = fluid_fraction_host > 0.0

    def measure_projected_divergence():
        multigrid_kernels["projected_divergence"](
            queue,
            (grid_cell_count,),
            None,
            pressure,
            rhs,
            fluid_fraction,
            *operator_level0,
            divergence_work,
            FIELD_DTYPE(dx),
            FIELD_DTYPE(dt / rho),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        cl.enqueue_copy(queue, divergence_host, divergence_work).wait()
        if not np.any(active_fluid):
            return 0.0
        values = divergence_host[active_fluid].astype(np.float64, copy=False)
        return float(np.sqrt(np.mean(values * values)))

    divergence_per_cycle = [measure_projected_divergence()]
    for _ in range(num_vcycles):
        multigrid.w_cycle(
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
            # The coarsest grid is tiny, so solving it substantially more
            # accurately is cheap and prevents its lowest mode from setting
            # the convergence rate of the whole multigrid cycle.
            coarse_smooth_iterations=64,
        )
        divergence_per_cycle.append(measure_projected_divergence())

    boundary_conditions.neumann_domain_bc(boundary_kernels, queue, pressure, grid_shape)
    return pressure, divergence_per_cycle


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
