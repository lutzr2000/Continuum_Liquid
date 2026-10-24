#include "helper.cl"

inline float domain_diagonal(const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    return 6.0f - (i == 1) - (i == nx - 2) - (j == 1) - (j == ny - 2) - (k == 1) - (k == nz - 2);
}

inline float weighted_neighbor_sum(__global const float *p,
                                   __global const float *weight_x,
                                   __global const float *weight_y,
                                   __global const float *weight_z,
                                   const int i,
                                   const int j,
                                   const int k,
                                   const int ny,
                                   const int nz) {
    const int index = grid_index(i, j, k, ny, nz);
    return weight_x[index] * p[grid_index(i + 1, j, k, ny, nz)] +
           weight_x[grid_index(i - 1, j, k, ny, nz)] * p[grid_index(i - 1, j, k, ny, nz)] +
           weight_y[index] * p[grid_index(i, j + 1, k, ny, nz)] +
           weight_y[grid_index(i, j - 1, k, ny, nz)] * p[grid_index(i, j - 1, k, ny, nz)] +
           weight_z[index] * p[grid_index(i, j, k + 1, ny, nz)] +
           weight_z[grid_index(i, j, k - 1, ny, nz)] * p[grid_index(i, j, k - 1, ny, nz)];
}

inline float apply_operator(__global const float *p,
                            __global const float *mask,
                            __global const float *diagonal,
                            __global const float *weight_x,
                            __global const float *weight_y,
                            __global const float *weight_z,
                            const float inv_delta2,
                            const int i,
                            const int j,
                            const int k,
                            const int nx,
                            const int ny,
                            const int nz) {
    const int index = grid_index(i, j, k, ny, nz);
    if (!is_interior_cell(i, j, k, nx, ny, nz) || mask[index] <= 0.0f)
        return 0.0f;
    return (weighted_neighbor_sum(p, weight_x, weight_y, weight_z, i, j, k, ny, nz) - diagonal[index] * p[index]) *
           inv_delta2;
}

inline float residual(__global const float *p,
                      __global const float *b,
                      __global const float *mask,
                      __global const float *diagonal,
                      __global const float *weight_x,
                      __global const float *weight_y,
                      __global const float *weight_z,
                      const float inv_delta2,
                      const int i,
                      const int j,
                      const int k,
                      const int nx,
                      const int ny,
                      const int nz) {
    const int index = grid_index(i, j, k, ny, nz);
    return b[index] - apply_operator(p, mask, diagonal, weight_x, weight_y, weight_z, inv_delta2, i, j, k, nx, ny, nz);
}

__kernel void build_finest_operator(__global const float *mask,
                                    __global float *diagonal,
                                    __global float *weight_x,
                                    __global float *weight_y,
                                    __global float *weight_z,
                                    const int nx,
                                    const int ny,
                                    const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;
    diagonal[index] = 0.0f;
    weight_x[index] = 0.0f;
    weight_y[index] = 0.0f;
    weight_z[index] = 0.0f;
    const int3 cell = grid_coordinates(index, ny, nz);
    if (!is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz) || mask[index] <= 0.0f)
        return;
    diagonal[index] = domain_diagonal(cell.x, cell.y, cell.z, nx, ny, nz);
    if (cell.x < nx - 2)
        weight_x[index] = mask[grid_index(cell.x + 1, cell.y, cell.z, ny, nz)] > 0.0f;
    if (cell.y < ny - 2)
        weight_y[index] = mask[grid_index(cell.x, cell.y + 1, cell.z, ny, nz)] > 0.0f;
    if (cell.z < nz - 2)
        weight_z[index] = mask[grid_index(cell.x, cell.y, cell.z + 1, ny, nz)] > 0.0f;
}

__kernel void restrict_residual(__global const float *fine_p,
                                __global const float *fine_b,
                                __global const float *fine_mask,
                                __global const float *fine_diagonal,
                                __global const float *fine_weight_x,
                                __global const float *fine_weight_y,
                                __global const float *fine_weight_z,
                                __global float *coarse_p,
                                __global float *coarse_b,
                                __global float *coarse_mask,
                                __global float *coarse_diagonal,
                                __global float *coarse_weight_x,
                                __global float *coarse_weight_y,
                                __global float *coarse_weight_z,
                                const float delta,
                                const int fine_nx,
                                const int fine_ny,
                                const int fine_nz,
                                const int coarse_nx,
                                const int coarse_ny,
                                const int coarse_nz) {
    const int coarse_index = get_global_id(0);
    if (coarse_index >= coarse_nx * coarse_ny * coarse_nz)
        return;
    coarse_p[coarse_index] = 0.0f;
    coarse_b[coarse_index] = 0.0f;
    coarse_mask[coarse_index] = 0.0f;
    coarse_diagonal[coarse_index] = 0.0f;
    coarse_weight_x[coarse_index] = 0.0f;
    coarse_weight_y[coarse_index] = 0.0f;
    coarse_weight_z[coarse_index] = 0.0f;

    const int3 coarse_cell = grid_coordinates(coarse_index, coarse_ny, coarse_nz);
    if (!is_interior_cell(coarse_cell.x, coarse_cell.y, coarse_cell.z, coarse_nx, coarse_ny, coarse_nz))
        return;
    const int3 fine_start = (int3)(1) + 2 * (coarse_cell - (int3)(1));
    const float inv_delta2 = 1.0f / (delta * delta);
    float rhs_sum = 0.0f;
    float diagonal_sum = 0.0f;
    float coupling_x = 0.0f;
    float coupling_y = 0.0f;
    float coupling_z = 0.0f;
    int active_count = 0;

    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const int3 cell = fine_start + (int3)(di, dj, dk);
                if (!is_interior_cell(cell.x, cell.y, cell.z, fine_nx, fine_ny, fine_nz))
                    continue;
                const int index = grid_index(cell.x, cell.y, cell.z, fine_ny, fine_nz);
                if (fine_mask[index] <= 0.0f)
                    continue;
                ++active_count;
                rhs_sum += residual(fine_p, fine_b, fine_mask, fine_diagonal, fine_weight_x, fine_weight_y,
                                    fine_weight_z, inv_delta2, cell.x, cell.y, cell.z, fine_nx, fine_ny, fine_nz);
                diagonal_sum += fine_diagonal[index];

                const float wx = fine_weight_x[index];
                const float wy = fine_weight_y[index];
                const float wz = fine_weight_z[index];
                if (wx > 0.0f) {
                    if (di == 0)
                        diagonal_sum -= 2.0f * wx;
                    else
                        coupling_x += wx;
                }
                if (wy > 0.0f) {
                    if (dj == 0)
                        diagonal_sum -= 2.0f * wy;
                    else
                        coupling_y += wy;
                }
                if (wz > 0.0f) {
                    if (dk == 0)
                        diagonal_sum -= 2.0f * wz;
                    else
                        coupling_z += wz;
                }
            }
        }
    }
    if (active_count > 0) {
        coarse_b[coarse_index] = rhs_sum;
        coarse_mask[coarse_index] = 1.0f;
        coarse_diagonal[coarse_index] = diagonal_sum;
        coarse_weight_x[coarse_index] = coupling_x;
        coarse_weight_y[coarse_index] = coupling_y;
        coarse_weight_z[coarse_index] = coupling_z;
    }
}

__kernel void prolongate(__global const float *coarse_error,
                         __global float *fine_p,
                         __global const float *fine_mask,
                         const int coarse_nx,
                         const int coarse_ny,
                         const int coarse_nz,
                         const int fine_nx,
                         const int fine_ny,
                         const int fine_nz) {
    const int fine_index = get_global_id(0);
    if (fine_index >= fine_nx * fine_ny * fine_nz)
        return;
    const int3 fine_cell = grid_coordinates(fine_index, fine_ny, fine_nz);
    if (!is_interior_cell(fine_cell.x, fine_cell.y, fine_cell.z, fine_nx, fine_ny, fine_nz) ||
        fine_mask[fine_index] <= 0.0f)
        return;
    const int3 coarse_cell =
        min((fine_cell - (int3)(1)) / 2 + (int3)(1), (int3)(coarse_nx - 2, coarse_ny - 2, coarse_nz - 2));
    fine_p[fine_index] += coarse_error[grid_index(coarse_cell.x, coarse_cell.y, coarse_cell.z, coarse_ny, coarse_nz)];
}

__kernel void rbgs_step(__global float *p,
                        __global const float *b,
                        __global const float *mask,
                        __global const float *diagonal,
                        __global const float *weight_x,
                        __global const float *weight_y,
                        __global const float *weight_z,
                        const float delta,
                        const int parity,
                        const int nx,
                        const int ny,
                        const int nz) {
    const int index = get_global_id(0);
    if (index >= nx * ny * nz)
        return;
    const int3 cell = grid_coordinates(index, ny, nz);
    if (!is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz) || mask[index] <= 0.0f ||
        ((cell.x + cell.y + cell.z) & 1) != parity) {
        if (mask[index] <= 0.0f)
            p[index] = 0.0f;
        return;
    }
    const float d = diagonal[index];
    if (d > 0.0f)
        p[index] = (weighted_neighbor_sum(p, weight_x, weight_y, weight_z, cell.x, cell.y, cell.z, ny, nz) -
                    delta * delta * b[index]) /
                   d;
}

__kernel void coarse_smooth(__global float *p,
                            __global const float *b,
                            __global const float *mask,
                            __global const float *diagonal,
                            __global const float *weight_x,
                            __global const float *weight_y,
                            __global const float *weight_z,
                            const float delta,
                            const int iterations,
                            const int nx,
                            const int ny,
                            const int nz) {
    const int thread_index = get_local_id(0);
    const int thread_count = get_local_size(0);
    const int cell_count = nx * ny * nz;
    const float delta2 = delta * delta;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int parity = 0; parity < 2; ++parity) {
            for (int index = thread_index; index < cell_count; index += thread_count) {
                const int3 cell = grid_coordinates(index, ny, nz);
                if (!is_interior_cell(cell.x, cell.y, cell.z, nx, ny, nz) || mask[index] <= 0.0f ||
                    ((cell.x + cell.y + cell.z) & 1) != parity)
                    continue;
                const float d = diagonal[index];
                if (d > 0.0f)
                    p[index] = (weighted_neighbor_sum(p, weight_x, weight_y, weight_z, cell.x, cell.y, cell.z, ny, nz) -
                                delta2 * b[index]) /
                               d;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
