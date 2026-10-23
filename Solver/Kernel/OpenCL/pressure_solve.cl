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
