#include "helper.cl"

__kernel void pressure_rhs(__global const float *u,
                           __global const float *v,
                           __global const float *w,
                           __global const float *fluid_fraction,
                           __global float *rhs,
                           const float rho_over_dt,
                           const float inv_dx,
                           const int nx,
                           const int ny,
                           const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    if (!is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz) || fluid_fraction[index] <= 0.0f) {
        rhs[index] = 0.0f;
        return;
    }

    rhs[index] = rho_over_dt * divergence(u, v, w, cell.x, cell.y, cell.z, inv_dx, ny, nz);
}

__kernel void pressure_solve(__global const float *pressure,
                             __global float *pressure_new,
                             __global const float *rhs,
                             __global const float *diagonal,
                             __global const float *fluid_fraction,
                             const float dx2,
                             const int nx,
                             const int ny,
                             const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    if (!is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz) || fluid_fraction[index] <= 0.0f) {
        pressure_new[index] = 0.0f;
        return;
    }

    float neighbor_sum = 0.0f;
    int neighbor_index = grid_index(cell.x + 1, cell.y, cell.z, ny, nz);
    neighbor_sum += pressure[neighbor_index] * (fluid_fraction[neighbor_index] > 0.0f);
    neighbor_index = grid_index(cell.x - 1, cell.y, cell.z, ny, nz);
    neighbor_sum += pressure[neighbor_index] * (fluid_fraction[neighbor_index] > 0.0f);
    neighbor_index = grid_index(cell.x, cell.y + 1, cell.z, ny, nz);
    neighbor_sum += pressure[neighbor_index] * (fluid_fraction[neighbor_index] > 0.0f);
    neighbor_index = grid_index(cell.x, cell.y - 1, cell.z, ny, nz);
    neighbor_sum += pressure[neighbor_index] * (fluid_fraction[neighbor_index] > 0.0f);
    neighbor_index = grid_index(cell.x, cell.y, cell.z + 1, ny, nz);
    neighbor_sum += pressure[neighbor_index] * (fluid_fraction[neighbor_index] > 0.0f);
    neighbor_index = grid_index(cell.x, cell.y, cell.z - 1, ny, nz);
    neighbor_sum += pressure[neighbor_index] * (fluid_fraction[neighbor_index] > 0.0f);

    const float cell_diagonal = diagonal[index];
    pressure_new[index] = cell_diagonal > 0.0f ? (neighbor_sum - rhs[index] * dx2) / cell_diagonal : 0.0f;
}

__kernel void pressure_correction(__global float *u,
                                  __global float *v,
                                  __global float *w,
                                  __global const float *pressure,
                                  const float dt_over_rho,
                                  const float inv_dx,
                                  const int nx,
                                  const int ny,
                                  const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;

    const int3 cell = grid_coordinates(index, ny, nz);
    if (!is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz))
        return;

    const float3 pressure_gradient = gradient(pressure, cell.x, cell.y, cell.z, inv_dx, ny, nz);
    u[index] -= dt_over_rho * pressure_gradient.x;
    v[index] -= dt_over_rho * pressure_gradient.y;
    w[index] -= dt_over_rho * pressure_gradient.z;
}
