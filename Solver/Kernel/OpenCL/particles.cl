#include "helper.cl"

inline int3 interpolation_base(const float3 position, const float inv_dx, const int nx, const int ny, const int nz) {
    const int3 base = convert_int3_rtn(position * inv_dx - 0.5f);
    return clamp(base, (int3)(0), (int3)(nx - 2, ny - 2, nz - 2));
}

inline float3 interpolation_fraction(const float3 position, const float inv_dx, const int3 base) {
    return clamp(position * inv_dx - 0.5f - convert_float3(base), 0.0f, 1.0f);
}

inline int3 staggered_interpolation_base(const float3 position,
                                         const float inv_dx,
                                         const float3 offset,
                                         const int nx,
                                         const int ny,
                                         const int nz) {
    const int3 base = convert_int3_rtn(position * inv_dx - offset);
    return clamp(base, (int3)(0), (int3)(nx - 2, ny - 2, nz - 2));
}

inline float3 staggered_interpolation_fraction(const float3 position,
                                               const float inv_dx,
                                               const float3 offset,
                                               const int3 base) {
    return clamp(position * inv_dx - offset - convert_float3(base), 0.0f, 1.0f);
}

inline float interpolation_weight(const float3 fraction, const int ox, const int oy, const int oz) {
    const float wx = ox ? fraction.x : 1.0f - fraction.x;
    const float wy = oy ? fraction.y : 1.0f - fraction.y;
    const float wz = oz ? fraction.z : 1.0f - fraction.z;
    return wx * wy * wz;
}

inline float
sample_field(__global const float *field, const int3 base, const float3 fraction, const int ny, const int nz) {
    float value = 0.0f;
    for (int ox = 0; ox <= 1; ++ox) {
        for (int oy = 0; oy <= 1; ++oy) {
            for (int oz = 0; oz <= 1; ++oz) {
                const int3 cell = base + (int3)(ox, oy, oz);
                value += interpolation_weight(fraction, ox, oy, oz) * field[grid_index(cell.x, cell.y, cell.z, ny, nz)];
            }
        }
    }
    return value;
}

__kernel void count_particles_per_cell(__global const float *particles,
                                       __global int *bin_counts,
                                       __global int *particle_count,
                                       const int number_of_particles,
                                       const float inv_dx,
                                       const int nx,
                                       const int ny,
                                       const int nz) {
    const int particle_index = get_global_id(0);
    if (particle_index >= number_of_particles)
        return;

    const int offset = 3 * particle_index;
    const float3 position = vload3(0, particles + offset);
    const int3 base = interpolation_base(position, inv_dx, nx, ny, nz);
    atomic_inc((volatile __global uint *)(bin_counts + grid_index(base.x, base.y, base.z, ny, nz)));

    const int3 cell = clamp(convert_int3_rtz(position * inv_dx), (int3)(1), (int3)(nx - 2, ny - 2, nz - 2));
    atomic_inc((volatile __global uint *)(particle_count + grid_index(cell.x, cell.y, cell.z, ny, nz)));
}

__kernel void scan_blocks_exclusive(
    __global const int *input, __global int *output, __global int *block_sums, const int count, __local int *scratch) {
    const int local_index = get_local_id(0);
    const int block_start = 2 * get_group_id(0) * get_local_size(0);
    const int first = block_start + local_index;
    const int second = first + get_local_size(0);
    const int scratch_count = 2 * get_local_size(0);

    scratch[local_index] = first < count ? input[first] : 0;
    scratch[local_index + get_local_size(0)] = second < count ? input[second] : 0;
    for (int offset = 1; offset < scratch_count; offset <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const int target = (local_index + 1) * offset * 2 - 1;
        if (target < scratch_count)
            scratch[target] += scratch[target - offset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        block_sums[get_group_id(0)] = scratch[scratch_count - 1];
        scratch[scratch_count - 1] = 0;
    }

    for (int offset = scratch_count >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const int target = (local_index + 1) * offset * 2 - 1;
        if (target < scratch_count) {
            const int left = scratch[target - offset];
            scratch[target - offset] = scratch[target];
            scratch[target] += left;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (first < count)
        output[first] = scratch[local_index];
    if (second < count)
        output[second] = scratch[local_index + get_local_size(0)];
}

__kernel void add_scanned_block_offsets(__global int *values,
                                        __global const int *block_offsets,
                                        const int count,
                                        const int elements_per_block) {
    const int index = get_global_id(0);
    if (index < count)
        values[index] += block_offsets[index / elements_per_block];
}

__kernel void sort_particles(__global const float *particles,
                             __global const float *particle_velocity,
                             __global float *particles_sorted,
                             __global float *particle_velocity_sorted,
                             __global const int *bin_offsets,
                             __global int *bin_cursors,
                             const int number_of_particles,
                             const float inv_dx,
                             const int nx,
                             const int ny,
                             const int nz) {
    const int particle_index = get_global_id(0);
    if (particle_index >= number_of_particles)
        return;

    const int source = 3 * particle_index;
    const float3 position = vload3(0, particles + source);
    const int3 base = interpolation_base(position, inv_dx, nx, ny, nz);
    const int bin_index = grid_index(base.x, base.y, base.z, ny, nz);
    const int local_index = atomic_inc((volatile __global uint *)(bin_cursors + bin_index));
    const int destination = 3 * (bin_offsets[bin_index] + local_index);
    vstore3(position, 0, particles_sorted + destination);
    vstore3(vload3(0, particle_velocity + source), 0, particle_velocity_sorted + destination);
}

__kernel void particle_to_grid_gather(__global const float *particles,
                                      __global const float *particle_velocity,
                                      __global const int *bin_offsets,
                                      __global const int *bin_counts,
                                      __global const int *particle_count,
                                      __global float *u,
                                      __global float *v,
                                      __global float *w,
                                      __global float *fluid_fraction,
                                      const float inv_dx,
                                      const float inv_particles_per_cell,
                                      const int nx,
                                      const int ny,
                                      const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    float3 velocity_sum = (float3)(0.0f);
    float3 weight_sum = (float3)(0.0f);

    // The pressure operators use a MAC layout.  u, v and w therefore live at
    // (i+1,j+.5,k+.5), (i+.5,j+1,k+.5), and (i+.5,j+.5,k+1).
    // Particle bins are cell-centred, so inspect the adjacent 3^3 bins and
    // evaluate each component's staggered trilinear weight explicitly.
    const float3 grid_position = convert_float3(cell);
    const float3 u_position = grid_position + (float3)(1.0f, 0.5f, 0.5f);
    const float3 v_position = grid_position + (float3)(0.5f, 1.0f, 0.5f);
    const float3 w_position = grid_position + (float3)(0.5f, 0.5f, 1.0f);
    for (int ox = -1; ox <= 1; ++ox) {
        const int base_x = cell.x + ox;
        if (base_x < 0 || base_x >= nx - 1)
            continue;
        for (int oy = -1; oy <= 1; ++oy) {
            const int base_y = cell.y + oy;
            if (base_y < 0 || base_y >= ny - 1)
                continue;
            for (int oz = -1; oz <= 1; ++oz) {
                const int base_z = cell.z + oz;
                if (base_z < 0 || base_z >= nz - 1)
                    continue;

                const int bin = grid_index(base_x, base_y, base_z, ny, nz);
                const int begin = bin_offsets[bin];
                const int end = begin + bin_counts[bin];
                for (int particle_index = begin; particle_index < end; ++particle_index) {
                    const int particle_offset = 3 * particle_index;
                    const float3 position = vload3(0, particles + particle_offset);
                    const float3 p = position * inv_dx;
                    const float3 wu_axis = max((float3)(0.0f), 1.0f - fabs(p - u_position));
                    const float3 wv_axis = max((float3)(0.0f), 1.0f - fabs(p - v_position));
                    const float3 ww_axis = max((float3)(0.0f), 1.0f - fabs(p - w_position));
                    const float3 weight =
                        (float3)(wu_axis.x * wu_axis.y * wu_axis.z,
                                 wv_axis.x * wv_axis.y * wv_axis.z,
                                 ww_axis.x * ww_axis.y * ww_axis.z);
                    velocity_sum += weight * vload3(0, particle_velocity + particle_offset);
                    weight_sum += weight;
                }
            }
        }
    }

    const float3 grid_velocity =
        (float3)(weight_sum.x > 0.0f ? velocity_sum.x / weight_sum.x : 0.0f,
                 weight_sum.y > 0.0f ? velocity_sum.y / weight_sum.y : 0.0f,
                 weight_sum.z > 0.0f ? velocity_sum.z / weight_sum.z : 0.0f);
    u[index] = grid_velocity.x;
    v[index] = grid_velocity.y;
    w[index] = grid_velocity.z;
    fluid_fraction[index] = convert_float(particle_count[index]) * inv_particles_per_cell;
}

__kernel void grid_to_particle(__global float *particle_velocity,
                               __global const float *particles,
                               __global const float *u,
                               __global const float *v,
                               __global const float *w,
                               __global const float *old_u,
                               __global const float *old_v,
                               __global const float *old_w,
                               const int number_of_particles,
                               const float inv_dx,
                               const int nx,
                               const int ny,
                               const int nz) {
    const int particle_index = get_global_id(0);
    if (particle_index >= number_of_particles)
        return;

    const int offset = 3 * particle_index;
    const float3 position = vload3(0, particles + offset);
    const float3 u_offset = (float3)(1.0f, 0.5f, 0.5f);
    const float3 v_offset = (float3)(0.5f, 1.0f, 0.5f);
    const float3 w_offset = (float3)(0.5f, 0.5f, 1.0f);
    const int3 u_base = staggered_interpolation_base(position, inv_dx, u_offset, nx, ny, nz);
    const int3 v_base = staggered_interpolation_base(position, inv_dx, v_offset, nx, ny, nz);
    const int3 w_base = staggered_interpolation_base(position, inv_dx, w_offset, nx, ny, nz);
    const float3 u_fraction = staggered_interpolation_fraction(position, inv_dx, u_offset, u_base);
    const float3 v_fraction = staggered_interpolation_fraction(position, inv_dx, v_offset, v_base);
    const float3 w_fraction = staggered_interpolation_fraction(position, inv_dx, w_offset, w_base);
    const float3 velocity_change =
        (float3)(sample_field(u, u_base, u_fraction, ny, nz) - sample_field(old_u, u_base, u_fraction, ny, nz),
                 sample_field(v, v_base, v_fraction, ny, nz) - sample_field(old_v, v_base, v_fraction, ny, nz),
                 sample_field(w, w_base, w_fraction, ny, nz) - sample_field(old_w, w_base, w_fraction, ny, nz));
    vstore3(vload3(0, particle_velocity + offset) + velocity_change, 0, particle_velocity + offset);
}

__kernel void advect_particles(__global float *particles,
                               __global const float *particle_velocity,
                               const int number_of_particles,
                               const float dt) {
    const int particle_index = get_global_id(0);
    if (particle_index >= number_of_particles)
        return;

    const int offset = 3 * particle_index;
    const float3 position = vload3(0, particles + offset);
    const float3 velocity = vload3(0, particle_velocity + offset);
    vstore3(position + dt * velocity, 0, particles + offset);
}
