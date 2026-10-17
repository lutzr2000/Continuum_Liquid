import numpy as np

from . import helper
from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def particle_to_grid(
    kernels,
    queue,
    particles,
    particle_velocity,
    velocity,
    weight,
    particle_count,
    fluid_fraction,
    particle_count_total,
    grid_shape,
    dx,
    particles_per_cell,
):
    nx, ny, nz = grid_shape
    u, v, w = velocity

    for buffer in (u, v, w, weight):
        helper.fill_device(queue, buffer, 0.0)
    helper.fill_device(queue, particle_count, 0, dtype=np.int32)

    kernels["particle_to_grid"](
        queue,
        (particle_count_total,),
        None,
        particles,
        particle_velocity,
        u,
        v,
        w,
        weight,
        particle_count,
        np.int32(particle_count_total),
        FIELD_DTYPE(1.0 / dx),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    kernels["normalize_particle_to_grid"](
        queue,
        (nx * ny * nz,),
        None,
        u,
        v,
        w,
        weight,
        particle_count,
        fluid_fraction,
        FIELD_DTYPE(1.0 / particles_per_cell),
        np.int32(nx * ny * nz),
    )
    return velocity, fluid_fraction


def grid_to_particle(
    kernels,
    queue,
    particle_velocity,
    particles,
    velocity,
    old_velocity,
    particle_count_total,
    grid_shape,
    dx,
):
    nx, ny, nz = grid_shape
    u, v, w = velocity
    old_u, old_v, old_w = old_velocity
    kernels["grid_to_particle"](
        queue,
        (particle_count_total,),
        None,
        particle_velocity,
        particles,
        u,
        v,
        w,
        old_u,
        old_v,
        old_w,
        np.int32(particle_count_total),
        FIELD_DTYPE(1.0 / dx),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    return particle_velocity


def advect_particles(
    kernels,
    queue,
    particles,
    particle_velocity,
    particle_count_total,
    dt,
):
    kernels["advect_particles"](
        queue,
        (particle_count_total,),
        None,
        particles,
        particle_velocity,
        np.int32(particle_count_total),
        FIELD_DTYPE(dt),
    )
    return particles
