#include "helper.cl"

__kernel void update_velocity(__global const float *u,
                              __global const float *v,
                              __global const float *w,
                              __global float *result_u,
                              __global float *result_v,
                              __global float *result_w,
                              __global const float *fluid_fraction,
                              const float gravity_x,
                              const float gravity_y,
                              const float gravity_z,
                              const float force_x,
                              const float force_y,
                              const float force_z,
                              const float rho,
                              const float nu,
                              const float inv_dx2,
                              const float dt,
                              const int nx,
                              const int ny,
                              const int nz) {
    const int index = get_global_id(0);
    const int cell_count = nx * ny * nz;
    if (index >= cell_count)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    float3 result = (float3)(u[index], v[index], w[index]);
    if (fluid_fraction[index] > 0.0f) {
        result += dt * ((float3)(gravity_x, gravity_y, gravity_z) + (float3)(force_x, force_y, force_z) / rho);
    }

    if (is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz)) {
        result.x += nu * dt * laplacian(u, cell.x, cell.y, cell.z, inv_dx2, ny, nz);
        result.y += nu * dt * laplacian(v, cell.x, cell.y, cell.z, inv_dx2, ny, nz);
        result.z += nu * dt * laplacian(w, cell.x, cell.y, cell.z, inv_dx2, ny, nz);
    }

    result_u[index] = result.x;
    result_v[index] = result.y;
    result_w[index] = result.z;
}
