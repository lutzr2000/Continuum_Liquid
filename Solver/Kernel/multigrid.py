from typing import Any

import numpy as np
import pyopencl as cl

from . import helper
from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def create_multigrid_levels(
    context: cl.Context,
    shape: tuple[int, int, int],
    delta: float,
    min_size: int = 8,
) -> Any:
    """Allocate the dense levels below the simulation grid."""
    p_levels = []
    b_levels = []
    mask_levels = []
    delta_levels = []
    level_shapes = []

    nx = (shape[0] + 1) // 2
    ny = (shape[1] + 1) // 2
    nz = (shape[2] + 1) // 2
    level = 1

    while nx >= min_size and ny >= min_size and nz >= min_size:
        level_shape = (nx, ny, nz)
        p_levels.append(helper.device_array(context, level_shape))
        b_levels.append(helper.device_array(context, level_shape))
        mask_levels.append(helper.device_array(context, level_shape))
        delta_levels.append(delta * (2**level))
        level_shapes.append(level_shape)

        nx = (nx + 1) // 2
        ny = (ny + 1) // 2
        nz = (nz + 1) // 2
        level += 1

    return p_levels, b_levels, mask_levels, delta_levels, level_shapes


def v_cycle(
    kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    level: int,
    p_levels: list[Any],
    b_levels: list[Any],
    mask_levels: list[Any],
    p_level0: Any,
    b_level0: Any,
    mask_level0: Any,
    base_delta: float,
    delta_levels: list[float],
    level_shapes: list[tuple[int, int, int]],
    level0_shape: tuple[int, int, int],
    pre_smooth: int,
    post_smooth: int,
    coarse_smooth_iterations: int,
) -> None:
    """Run one dense multigrid V-cycle for ``laplace(p) = b``."""
    if level == 0:
        p, b, mask = p_level0, b_level0, mask_level0
        delta = base_delta
        shape = level0_shape
    else:
        current = level - 1
        p, b, mask = p_levels[current], b_levels[current], mask_levels[current]
        delta = delta_levels[current]
        shape = level_shapes[current]

    if level == len(p_levels):
        coarse_smooth(
            kernels,
            queue,
            p,
            b,
            mask,
            delta,
            pre_smooth + coarse_smooth_iterations,
            shape,
        )
        return

    smooth(kernels, queue, p, b, mask, delta, pre_smooth, shape)

    coarse = level
    coarse_p = p_levels[coarse]
    coarse_b = b_levels[coarse]
    coarse_mask = mask_levels[coarse]
    coarse_shape = level_shapes[coarse]

    kernels["restrict_residual"](
        queue,
        (int(np.prod(coarse_shape)),),
        None,
        p,
        b,
        mask,
        coarse_p,
        coarse_b,
        coarse_mask,
        FIELD_DTYPE(delta),
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
        p_level0,
        b_level0,
        mask_level0,
        base_delta,
        delta_levels,
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

    smooth(kernels, queue, p, b, mask, delta, post_smooth, shape)


def smooth(
    kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    p: Any,
    b: Any,
    mask: Any,
    delta: float,
    iterations: int,
    shape: tuple[int, int, int],
) -> None:
    """Apply red-black Gauss-Seidel smoothing to one dense level."""
    cell_count = int(np.prod(shape))
    args = (
        p,
        b,
        mask,
        FIELD_DTYPE(delta),
        np.int32(shape[0]),
        np.int32(shape[1]),
        np.int32(shape[2]),
    )
    kernel = kernels["rbgs_step"]

    for _ in range(iterations):
        kernel(queue, (cell_count,), None, *args[:4], np.int32(0), *args[4:])
        kernel(queue, (cell_count,), None, *args[:4], np.int32(1), *args[4:])


def coarse_smooth(
    kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    p: Any,
    b: Any,
    mask: Any,
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
        FIELD_DTYPE(delta),
        np.int32(iterations),
        np.int32(shape[0]),
        np.int32(shape[1]),
        np.int32(shape[2]),
    )
