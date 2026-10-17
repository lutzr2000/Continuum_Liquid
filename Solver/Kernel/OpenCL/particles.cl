#include "helper.cl"

inline void atomic_add_float(volatile __global float *address, const float value) {
    volatile __global uint *integer_address = (volatile __global uint *)address;
    uint old_value = *integer_address;
    while (1) {
        const uint expected = old_value;
        const uint desired = as_uint(as_float(expected) + value);
        old_value = atomic_cmpxchg(integer_address, expected, desired);
        if (old_value == expected)
            return;
    }
}

inline int3 interpolation_base(const float3 position, const float inv_dx, const int nx, const int ny, const int nz) {
    const int3 base = convert_int3_rtn(position * inv_dx - 0.5f);
    return clamp(base, (int3)(0), (int3)(nx - 2, ny - 2, nz - 2));
}

inline float3 interpolation_fraction(const float3 position, const float inv_dx, const int3 base) {
    return clamp(position * inv_dx - 0.5f - convert_float3(base), 0.0f, 1.0f);
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

__kernel void particle_to_grid(__global const float *particles,
                               __global const float *particle_velocity,
                               __global float *u,
                               __global float *v,
                               __global float *w,
                               __global float *weight,
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
    const float3 particle_v = vload3(0, particle_velocity + offset);
    const int3 base = interpolation_base(position, inv_dx, nx, ny, nz);
    const float3 fraction = interpolation_fraction(position, inv_dx, base);

    for (int ox = 0; ox <= 1; ++ox) {
        for (int oy = 0; oy <= 1; ++oy) {
            for (int oz = 0; oz <= 1; ++oz) {
                const int3 cell = base + (int3)(ox, oy, oz);
                const int index = grid_index(cell.x, cell.y, cell.z, ny, nz);
                const float cell_weight = interpolation_weight(fraction, ox, oy, oz);
                atomic_add_float(u + index, cell_weight * particle_v.x);
                atomic_add_float(v + index, cell_weight * particle_v.y);
                atomic_add_float(w + index, cell_weight * particle_v.z);
                atomic_add_float(weight + index, cell_weight);
            }
        }
    }

    const int3 cell = clamp(convert_int3_rtz(position * inv_dx), (int3)(1), (int3)(nx - 2, ny - 2, nz - 2));
    atomic_inc((volatile __global uint *)(particle_count + grid_index(cell.x, cell.y, cell.z, ny, nz)));
}

__kernel void normalize_particle_to_grid(__global float *u,
                                         __global float *v,
                                         __global float *w,
                                         __global const float *weight,
                                         __global const int *particle_count,
                                         __global float *fluid_fraction,
                                         const float inv_particles_per_cell,
                                         const int grid_cell_count) {
    const int index = get_global_id(0);
    if (index >= grid_cell_count)
        return;

    const float cell_weight = weight[index];
    if (cell_weight > 0.0f) {
        const float inv_weight = 1.0f / cell_weight;
        u[index] *= inv_weight;
        v[index] *= inv_weight;
        w[index] *= inv_weight;
    }
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
    const int3 base = interpolation_base(position, inv_dx, nx, ny, nz);
    const float3 fraction = interpolation_fraction(position, inv_dx, base);
    const float3 velocity_change =
        (float3)(sample_field(u, base, fraction, ny, nz) - sample_field(old_u, base, fraction, ny, nz),
                 sample_field(v, base, fraction, ny, nz) - sample_field(old_v, base, fraction, ny, nz),
                 sample_field(w, base, fraction, ny, nz) - sample_field(old_w, base, fraction, ny, nz));
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
