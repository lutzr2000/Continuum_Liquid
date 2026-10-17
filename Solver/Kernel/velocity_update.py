import numpy as np

from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def update_velocity(
    kernels,
    queue,
    velocity,
    result,
    fluid_fraction,
    gravity,
    fx,
    fy,
    fz,
    rho,
    nu,
    dx,
    dt,
    grid_shape,
):
    nx, ny, nz = grid_shape
    u, v, w = velocity
    result_u, result_v, result_w = result
    kernels["update_velocity"](
        queue,
        (nx * ny * nz,),
        None,
        u,
        v,
        w,
        result_u,
        result_v,
        result_w,
        fluid_fraction,
        FIELD_DTYPE(gravity[0]),
        FIELD_DTYPE(gravity[1]),
        FIELD_DTYPE(gravity[2]),
        FIELD_DTYPE(fx),
        FIELD_DTYPE(fy),
        FIELD_DTYPE(fz),
        FIELD_DTYPE(rho),
        FIELD_DTYPE(nu),
        FIELD_DTYPE(1.0 / (dx * dx)),
        FIELD_DTYPE(dt),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    return result
