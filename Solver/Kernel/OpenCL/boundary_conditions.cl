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

__kernel void extrapolate_obstacle_velocity(__global float *u,
                                            __global float *v,
                                            __global float *w,
                                            __global const uchar *obstacle,
                                            const int nx,
                                            const int ny,
                                            const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz || !obstacle[index])
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    float3 tangential_velocity = (float3)(0.0f);
    int fluid_neighbours = 0;
    int3 surface_axis = (int3)(0);

#define ACCUMULATE_FLUID_NEIGHBOUR(i, j, k, axis)                                    \
    {                                                                                 \
        const int neighbour_index = grid_index((i), (j), (k), ny, nz);                \
        if (!obstacle[neighbour_index]) {                                              \
            tangential_velocity +=                                                    \
                (float3)(u[neighbour_index], v[neighbour_index], w[neighbour_index]);  \
            fluid_neighbours += 1;                                                     \
            surface_axis.axis = 1;                                                     \
        }                                                                              \
    }

    if (cell.x > 0)
        ACCUMULATE_FLUID_NEIGHBOUR(cell.x - 1, cell.y, cell.z, x);
    if (cell.x + 1 < nx)
        ACCUMULATE_FLUID_NEIGHBOUR(cell.x + 1, cell.y, cell.z, x);
    if (cell.y > 0)
        ACCUMULATE_FLUID_NEIGHBOUR(cell.x, cell.y - 1, cell.z, y);
    if (cell.y + 1 < ny)
        ACCUMULATE_FLUID_NEIGHBOUR(cell.x, cell.y + 1, cell.z, y);
    if (cell.z > 0)
        ACCUMULATE_FLUID_NEIGHBOUR(cell.x, cell.y, cell.z - 1, z);
    if (cell.z + 1 < nz)
        ACCUMULATE_FLUID_NEIGHBOUR(cell.x, cell.y, cell.z + 1, z);

#undef ACCUMULATE_FLUID_NEIGHBOUR

    if (fluid_neighbours == 0)
        return;

    tangential_velocity /= (float)fluid_neighbours;
    // Match the domain-wall ghost cells: keep the component normal to every
    // exposed surface at zero, but extend tangential flow into the solid cell
    // so interpolation next to the obstacle does not artificially damp it.
    u[index] = surface_axis.x ? 0.0f : tangential_velocity.x;
    v[index] = surface_axis.y ? 0.0f : tangential_velocity.y;
    w[index] = surface_axis.z ? 0.0f : tangential_velocity.z;
}

__kernel void particle_obstacle_bc(__global float *particles,
                                   __global const float *previous_particles,
                                   __global float *particle_velocity,
                                   __global const uchar *obstacle,
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
    const float3 previous = vload3(0, previous_particles + offset);
    float3 velocity = vload3(0, particle_velocity + offset);
    const int3 cell = clamp(convert_int3_rtz(position * inv_dx),
                            (int3)(0), (int3)(nx - 1, ny - 1, nz - 1));
    if (!obstacle[grid_index(cell.x, cell.y, cell.z, ny, nz)])
        return;

    // Estimate the outward surface normal from the empty cells adjacent to
    // the solid voxel.  Removing only velocity into the obstacle preserves
    // tangential motion, allowing particles to slide instead of sticking.
    const int3 previous_cell = clamp(convert_int3_rtz(previous * inv_dx),
                                     (int3)(0), (int3)(nx - 1, ny - 1, nz - 1));
    float3 normal = (float3)(0.0f);
    if (any(previous_cell != cell) &&
        !obstacle[grid_index(previous_cell.x, previous_cell.y, previous_cell.z, ny, nz)]) {
        // Prefer the face through which the particle entered.  This also
        // avoids cancelling normals on obstacles only one voxel thick.
        normal = convert_float3(previous_cell - cell);
    } else {
        if (cell.x > 0 && !obstacle[grid_index(cell.x - 1, cell.y, cell.z, ny, nz)])
            normal.x -= 1.0f;
        if (cell.x + 1 < nx && !obstacle[grid_index(cell.x + 1, cell.y, cell.z, ny, nz)])
            normal.x += 1.0f;
        if (cell.y > 0 && !obstacle[grid_index(cell.x, cell.y - 1, cell.z, ny, nz)])
            normal.y -= 1.0f;
        if (cell.y + 1 < ny && !obstacle[grid_index(cell.x, cell.y + 1, cell.z, ny, nz)])
            normal.y += 1.0f;
        if (cell.z > 0 && !obstacle[grid_index(cell.x, cell.y, cell.z - 1, ny, nz)])
            normal.z -= 1.0f;
        if (cell.z + 1 < nz && !obstacle[grid_index(cell.x, cell.y, cell.z + 1, ny, nz)])
            normal.z += 1.0f;
    }

    const float normal_length = length(normal);
    if (normal_length > 0.0f) {
        normal /= normal_length;
    } else {
        // This is mainly a safeguard for particles that crossed more than one
        // voxel in a step and ended inside a fully surrounded solid cell.
        const float3 fallback = previous - position;
        const float fallback_length = length(fallback);
        normal = fallback_length > 0.0f
                     ? fallback / fallback_length
                     : (float3)(0.0f, 0.0f, 1.0f);
    }

    const float normal_velocity = dot(velocity, normal);
    if (normal_velocity < 0.0f)
        velocity -= normal_velocity * normal;

    // Preserve the overshoot just like particle_domain_bc does.  Reflect each
    // crossed normal axis at the voxel face instead of collapsing all hits
    // back onto their previous positions.
    const float dx = 1.0f / inv_dx;
    float3 corrected = position;
    if (normal.x < 0.0f)
        corrected.x = 2.0f * ((float)cell.x * dx) - position.x;
    else if (normal.x > 0.0f)
        corrected.x = 2.0f * ((float)(cell.x + 1) * dx) - position.x;
    if (normal.y < 0.0f)
        corrected.y = 2.0f * ((float)cell.y * dx) - position.y;
    else if (normal.y > 0.0f)
        corrected.y = 2.0f * ((float)(cell.y + 1) * dx) - position.y;
    if (normal.z < 0.0f)
        corrected.z = 2.0f * ((float)cell.z * dx) - position.z;
    else if (normal.z > 0.0f)
        corrected.z = 2.0f * ((float)(cell.z + 1) * dx) - position.z;

    const int3 corrected_cell = clamp(convert_int3_rtz(corrected * inv_dx),
                                       (int3)(0), (int3)(nx - 1, ny - 1, nz - 1));
    if (obstacle[grid_index(corrected_cell.x, corrected_cell.y, corrected_cell.z, ny, nz)])
        corrected = previous;

    vstore3(corrected, 0, particles + offset);
    vstore3(velocity, 0, particle_velocity + offset);
}
