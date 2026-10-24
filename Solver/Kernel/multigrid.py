from typing import Any

import numpy as np
import pyopencl as cl

from . import helper
from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def coarse_grid_size(fine_size: int) -> int:
    """Coarsen cell centres while retaining one ghost cell on either side."""
    interior_size = max(0, fine_size - 2)
    return (interior_size + 1) // 2 + 2


def create_multigrid_levels(
    context: cl.Context,
    shape: tuple[int, int, int],
    min_size: int = 8,
) -> Any:
    """Allocate the dense levels below the simulation grid."""
    p_levels = []
    b_levels = []
    mask_levels = []
    operator_levels = []
    level_shapes = []

    nx = coarse_grid_size(shape[0])
    ny = coarse_grid_size(shape[1])
    nz = coarse_grid_size(shape[2])
    level = 1

    while nx >= min_size and ny >= min_size and nz >= min_size:
        level_shape = (nx, ny, nz)
        p_levels.append(helper.device_array(context, level_shape))
        b_levels.append(helper.device_array(context, level_shape))
        mask_levels.append(helper.device_array(context, level_shape))
        operator_levels.append(tuple(helper.device_array(context, level_shape) for _ in range(4)))
        level_shapes.append(level_shape)

        nx = coarse_grid_size(nx)
        ny = coarse_grid_size(ny)
        nz = coarse_grid_size(nz)
        level += 1

    return p_levels, b_levels, mask_levels, operator_levels, level_shapes


def v_cycle(
    kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    level: int,
    p_levels: list[Any],
    b_levels: list[Any],
    mask_levels: list[Any],
    operator_levels: list[tuple[Any, Any, Any, Any]],
    p_level0: Any,
    b_level0: Any,
    mask_level0: Any,
    operator_level0: tuple[Any, Any, Any, Any],
    base_delta: float,
    level_shapes: list[tuple[int, int, int]],
    level0_shape: tuple[int, int, int],
    pre_smooth: int,
    post_smooth: int,
    coarse_smooth_iterations: int,
) -> None:
    """Run one dense multigrid V-cycle for ``laplace(p) = b``."""
    if level == 0:
        p, b, mask = p_level0, b_level0, mask_level0
        operator = operator_level0
        shape = level0_shape
    else:
        current = level - 1
        p, b, mask = p_levels[current], b_levels[current], mask_levels[current]
        operator = operator_levels[current]
        shape = level_shapes[current]

    if level == len(p_levels):
        coarse_smooth(
            kernels,
            queue,
            p,
            b,
            mask,
            operator,
            base_delta,
            pre_smooth + coarse_smooth_iterations,
            shape,
        )
        return

    smooth(kernels, queue, p, b, mask, operator, base_delta, pre_smooth, shape)

    coarse = level
    coarse_p = p_levels[coarse]
    coarse_b = b_levels[coarse]
    coarse_mask = mask_levels[coarse]
    coarse_operator = operator_levels[coarse]
    coarse_shape = level_shapes[coarse]

    kernels["restrict_residual"](
        queue,
        (int(np.prod(coarse_shape)),),
        None,
        p,
        b,
        mask,
        *operator,
        coarse_p,
        coarse_b,
        coarse_mask,
        *coarse_operator,
        FIELD_DTYPE(base_delta),
        np.int32(shape[0]),
        np.int32(shape[1]),
        np.int32(shape[2]),
        np.int32(coarse_shape[0]),
        np.int32(coarse_shape[1]),
        np.int32(coarse_shape[2]),
    )

    v_cycle(
        kernels,
        queue,
        level + 1,
        p_levels,
        b_levels,
        mask_levels,
        operator_levels,
        p_level0,
        b_level0,
        mask_level0,
        operator_level0,
        base_delta,
        level_shapes,
        level0_shape,
        pre_smooth,
        post_smooth,
        coarse_smooth_iterations,
    )

    kernels["prolongate"](
        queue,
        (int(np.prod(shape)),),
        None,
        coarse_p,
        p,
        mask,
        np.int32(coarse_shape[0]),
        np.int32(coarse_shape[1]),
        np.int32(coarse_shape[2]),
        np.int32(shape[0]),
        np.int32(shape[1]),
        np.int32(shape[2]),
    )

    smooth(kernels, queue, p, b, mask, operator, base_delta, post_smooth, shape)


def smooth(
    kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    p: Any,
    b: Any,
    mask: Any,
    operator: tuple[Any, Any, Any, Any],
    delta: float,
    iterations: int,
    shape: tuple[int, int, int],
) -> None:
    """Apply red-black Gauss-Seidel smoothing to one dense level."""
    cell_count = int(np.prod(shape))
    kernel = kernels["rbgs_step"]

    for _ in range(iterations):
        for parity in (0, 1):
            kernel(
                queue,
                (cell_count,),
                None,
                p,
                b,
                mask,
                *operator,
                FIELD_DTYPE(delta),
                np.int32(parity),
                np.int32(shape[0]),
                np.int32(shape[1]),
                np.int32(shape[2]),
            )


def coarse_smooth(
    kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    p: Any,
    b: Any,
    mask: Any,
    operator: tuple[Any, Any, Any, Any],
    delta: float,
    iterations: int,
    shape: tuple[int, int, int],
) -> None:
    """Smooth the coarsest level in one persistent kernel launch."""
    threads = kernel_config.REDUCTION_THREADS_PER_BLOCK
    kernels["coarse_smooth"](
        queue,
        (threads,),
        (threads,),
        p,
        b,
        mask,
        *operator,
        FIELD_DTYPE(delta),
        np.int32(iterations),
        np.int32(shape[0]),
        np.int32(shape[1]),
        np.int32(shape[2]),
    )
