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
    const int ghost = !is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz);

    // u[i] is the x-face on the positive side of pressure cell i (and likewise
    // for v and w).  Consequently the upper solid face is nx-2, not nx-1.
    // The old boundary condition only cleared the padding cell and left this
    // actual wall face open.
    const int wall_u = cell.x == 0 || cell.x >= nx - 2;
    const int wall_v = cell.y == 0 || cell.y >= ny - 2;
    const int wall_w = cell.z == 0 || cell.z >= nz - 2;
    if (!ghost && !wall_u && !wall_v && !wall_w)
        return;

    if (ghost) {
        const int3 interior = clamp(cell, (int3)(1), (int3)(nx - 2, ny - 2, nz - 2));
        const int interior_index = grid_index(interior.x, interior.y, interior.z, ny, nz);
        u[index] = wall_u ? 0.0f : u[interior_index];
        v[index] = wall_v ? 0.0f : v[interior_index];
        w[index] = wall_w ? 0.0f : w[interior_index];
        return;
    }

    if (wall_u)
        u[index] = 0.0f;
    if (wall_v)
        v[index] = 0.0f;
    if (wall_w)
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

    // Preserve the overshoot distance instead of snapping every colliding
    // particle onto the exact same wall plane.  This avoids collapsing the
    // particle distribution into dense lines where two walls meet.
    position = select(position, 2.0f * lower - position, hit_lower);
    position = select(position, 2.0f * upper - position, hit_upper);
    position = clamp(position, (float3)(lower), upper);
    velocity = select(velocity, (float3)(0.0f), hit);
    vstore3(position, 0, particles + offset);
    vstore3(velocity, 0, particle_velocity + offset);
}

__kernel void grid_obstacle_bc(__global float *u,
                               __global float *v,
                               __global float *w,
                               __global float *fluid_fraction,
                               __global const uchar *obstacle,
                               const int nx,
                               const int ny,
                               const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;
    const int3 cell = grid_coordinates(index, ny, nz);
    if (obstacle[index])
        fluid_fraction[index] = 0.0f;
    const int x_solid = obstacle[index] || (cell.x + 1 < nx && obstacle[grid_index(cell.x + 1, cell.y, cell.z, ny, nz)]);
    const int y_solid = obstacle[index] || (cell.y + 1 < ny && obstacle[grid_index(cell.x, cell.y + 1, cell.z, ny, nz)]);
    const int z_solid = obstacle[index] || (cell.z + 1 < nz && obstacle[grid_index(cell.x, cell.y, cell.z + 1, ny, nz)]);
    if (x_solid) u[index] = 0.0f;
    if (y_solid) v[index] = 0.0f;
    if (z_solid) w[index] = 0.0f;
}
