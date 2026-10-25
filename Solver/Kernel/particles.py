import numpy as np
import pyopencl as cl

from . import helper
from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def create_particle_sort_buffers(context, grid_shape, particle_count_total):
    """Allocate double buffers and scratch storage for cell sorting."""
    cell_count = int(np.prod(grid_shape))
    elements_per_block = 2 * kernel_config.REDUCTION_THREADS_PER_BLOCK

    particle_work = helper.device_array(context, (particle_count_total, 3))
    particle_velocity_work = helper.device_array(context, (particle_count_total, 3))
    bin_counts = helper.device_array(context, cell_count, dtype=np.int32)
    scan_outputs = [helper.device_array(context, cell_count, dtype=np.int32)]
    scan_sums = []

    level_size = cell_count
    while True:
        block_count = (level_size + elements_per_block - 1) // elements_per_block
        scan_sums.append(helper.device_array(context, block_count, dtype=np.int32))
        if block_count == 1:
            break
        scan_outputs.append(helper.device_array(context, block_count, dtype=np.int32))
        level_size = block_count

    return (
        particle_work,
        particle_velocity_work,
        bin_counts,
        scan_outputs,
        scan_sums,
    )


def exclusive_prefix_sum(kernels, queue, values, outputs, sums, count):
    """Compute an exclusive integer scan entirely on the OpenCL device."""
    threads = kernel_config.REDUCTION_THREADS_PER_BLOCK
    elements_per_block = 2 * threads
    inputs = [values, *sums[:-1]]
    level_count = count

    for level, output in enumerate(outputs):
        block_count = (level_count + elements_per_block - 1) // elements_per_block
        kernels["scan_blocks_exclusive"](
            queue,
            (block_count * threads,),
            (threads,),
            inputs[level],
            output,
            sums[level],
            np.int32(level_count),
            cl.LocalMemory(elements_per_block * np.dtype(np.int32).itemsize),
        )
        level_count = block_count

    for level in range(len(outputs) - 2, -1, -1):
        level_count = outputs[level].size // np.dtype(np.int32).itemsize
        kernels["add_scanned_block_offsets"](
            queue,
            (level_count,),
            None,
            outputs[level],
            outputs[level + 1],
            np.int32(level_count),
            np.int32(elements_per_block),
        )


def particle_to_grid(
    kernels,
    queue,
    particles,
    particle_velocity,
    particle_work,
    particle_velocity_work,
    velocity,
    particle_bin_counts,
    particle_bin_scan_outputs,
    particle_bin_scan_sums,
    fluid_fraction,
    particle_count_total,
    grid_shape,
    dx,
    particles_per_cell,
):
    nx, ny, nz = grid_shape
    u, v, w = velocity
    grid_cell_count = nx * ny * nz

    helper.fill_device(queue, particle_bin_counts, 0, dtype=np.int32)

    kernels["count_particles_per_cell"](
        queue,
        (particle_count_total,),
        None,
        particles,
        particle_bin_counts,
        np.int32(particle_count_total),
        FIELD_DTYPE(1.0 / dx),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )

    exclusive_prefix_sum(
        kernels,
        queue,
        particle_bin_counts,
        particle_bin_scan_outputs,
        particle_bin_scan_sums,
        grid_cell_count,
    )

    particle_bin_offsets = particle_bin_scan_outputs[0]
    helper.fill_device(queue, particle_bin_counts, 0, dtype=np.int32)
    kernels["sort_particles"](
        queue,
        (particle_count_total,),
        None,
        particles,
        particle_velocity,
        particle_work,
        particle_velocity_work,
        particle_bin_offsets,
        particle_bin_counts,
        np.int32(particle_count_total),
        FIELD_DTYPE(1.0 / dx),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
    )
    particles, particle_work = particle_work, particles
    particle_velocity, particle_velocity_work = (
        particle_velocity_work,
        particle_velocity,
    )

    gather_threads = kernel_config.GATHER_THREADS_PER_CELL
    kernels["particle_to_grid_gather"](
        queue,
        (grid_cell_count * gather_threads,),
        (gather_threads,),
        particles,
        particle_velocity,
        particle_bin_offsets,
        particle_bin_counts,
        u,
        v,
        w,
        fluid_fraction,
        FIELD_DTYPE(1.0 / dx),
        FIELD_DTYPE(1.0 / particles_per_cell),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        cl.LocalMemory(gather_threads * 4 * np.dtype(FIELD_DTYPE).itemsize),
        cl.LocalMemory(gather_threads * 4 * np.dtype(FIELD_DTYPE).itemsize),
        cl.LocalMemory(gather_threads * np.dtype(np.int32).itemsize),
    )
    return (
        particles,
        particle_velocity,
        particle_work,
        particle_velocity_work,
        velocity,
        fluid_fraction,
    )


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
