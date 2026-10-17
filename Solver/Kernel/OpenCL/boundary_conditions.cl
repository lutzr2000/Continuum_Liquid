#include "helper.cl"

__kernel void create_pressure_boundary(__global float *diagonal, const int nx, const int ny, const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    if (!is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz)) {
        diagonal[index] = 0.0f;
        return;
    }

    diagonal[index] = 6.0f - (cell.x == 1) - (cell.x == nx - 2) - (cell.y == 1) - (cell.y == ny - 2) - (cell.z == 1) -
                      (cell.z == nz - 2);
}

__kernel void neumann_domain_bc(__global float *pressure, const int nx, const int ny, const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    if (is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz))
        return;

    const int i = clamp(cell.x, 1, nx - 2);
    const int j = clamp(cell.y, 1, ny - 2);
    const int k = clamp(cell.z, 1, nz - 2);
    pressure[index] = pressure[grid_index(i, j, k, ny, nz)];
}

__kernel void
grid_domain_bc(__global float *u, __global float *v, __global float *w, const int nx, const int ny, const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    if (is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz))
        return;

    u[index] = 0.0f;
    v[index] = 0.0f;
    w[index] = 0.0f;
}

__kernel void particle_domain_bc(__global float *particles,
                                 __global float *particle_velocity,
                                 const int number_of_particles,
                                 const float lower,
                                 const float upper_x,
                                 const float upper_y,
                                 const float upper_z) {
    const int particle_index = get_global_id(0);
    if (particle_index >= number_of_particles)
        return;

    const int offset = 3 * particle_index;
    float3 position = vload3(0, particles + offset);
    float3 velocity = vload3(0, particle_velocity + offset);
    const float3 upper = (float3)(upper_x, upper_y, upper_z);
    const int3 hit_lower = position < lower;
    const int3 hit_upper = position > upper;
    const int3 hit = hit_lower | hit_upper;

    position = clamp(position, (float3)(lower), upper);
    velocity = select(velocity, (float3)(0.0f), hit);
    vstore3(position, 0, particles + offset);
    vstore3(velocity, 0, particle_velocity + offset);
}
