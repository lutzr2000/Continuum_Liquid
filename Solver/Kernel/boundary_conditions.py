import numpy as np

from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def create_pressure_boundary(kernels, queue, diagonal, grid_shape):
    nx, ny, nz = grid_shape
    kernels["create_pressure_boundary"](
        queue,
        (nx * ny * nz,),
        None,
        diagonal,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    return diagonal


def neumann_domain_bc(kernels, queue, pressure, grid_shape):
    nx, ny, nz = grid_shape
    kernels["neumann_domain_bc"](
        queue,
        (nx * ny * nz,),
        None,
        pressure,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    return pressure


def grid_domain_bc(kernels, queue, velocity, grid_shape):
    nx, ny, nz = grid_shape
    u, v, w = velocity
    kernels["grid_domain_bc"](
        queue,
        (nx * ny * nz,),
        None,
        u,
        v,
        w,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    return velocity


def grid_obstacle_bc(kernels, queue, velocity, fluid_fraction, obstacle, grid_shape):
    nx, ny, nz = grid_shape
    kernels["grid_obstacle_bc"](
        queue,
        (nx * ny * nz,),
        None,
        *velocity,
        fluid_fraction,
        obstacle,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    return velocity, fluid_fraction


def particle_domain_bc(
    kernels,
    queue,
    particles,
    particle_velocity,
    particle_count_total,
    grid_shape,
    dx,
):
    kernels["particle_domain_bc"](
        queue,
        (particle_count_total,),
        None,
        particles,
        particle_velocity,
        np.int32(particle_count_total),
        FIELD_DTYPE(1.001 * dx),
        FIELD_DTYPE((grid_shape[0] - 1.001) * dx),
        FIELD_DTYPE((grid_shape[1] - 1.001) * dx),
        FIELD_DTYPE((grid_shape[2] - 1.001) * dx),
    )
    return particles, particle_velocity
