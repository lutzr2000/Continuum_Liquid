#include "helper.cl"

inline float dense_diagonal(const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    return 6.0f - (i == 1) - (i == nx - 2) - (j == 1) - (j == ny - 2) - (k == 1) - (k == nz - 2);
}

inline float active_neighbor_sum(__global const float *p,
                                 __global const float *mask,
                                 const int i,
                                 const int j,
                                 const int k,
                                 const int ny,
                                 const int nz) {
    int neighbor = grid_index(i + 1, j, k, ny, nz);
    float sum = p[neighbor] * (mask[neighbor] > 0.0f);
    neighbor = grid_index(i - 1, j, k, ny, nz);
    sum += p[neighbor] * (mask[neighbor] > 0.0f);
    neighbor = grid_index(i, j + 1, k, ny, nz);
    sum += p[neighbor] * (mask[neighbor] > 0.0f);
    neighbor = grid_index(i, j - 1, k, ny, nz);
    sum += p[neighbor] * (mask[neighbor] > 0.0f);
    neighbor = grid_index(i, j, k + 1, ny, nz);
    sum += p[neighbor] * (mask[neighbor] > 0.0f);
    neighbor = grid_index(i, j, k - 1, ny, nz);
    sum += p[neighbor] * (mask[neighbor] > 0.0f);
    return sum;
}

inline float residual(__global const float *p,
                      __global const float *b,
                      __global const float *mask,
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

    const float laplace =
        (active_neighbor_sum(p, mask, i, j, k, ny, nz) - dense_diagonal(i, j, k, nx, ny, nz) * p[index]) * inv_delta2;
    return b[index] - laplace;
}

__kernel void restrict_residual(__global const float *fine_p,
                                __global const float *fine_b,
                                __global const float *fine_mask,
                                __global float *coarse_p,
                                __global float *coarse_b,
                                __global float *coarse_mask,
                                const float fine_delta,
                                const int fine_nx,
                                const int fine_ny,
                                const int fine_nz,
                                const int coarse_nx,
                                const int coarse_ny,
                                const int coarse_nz) {
    const int coarse_index = get_global_id(0);
    if (coarse_index >= coarse_nx * coarse_ny * coarse_nz)
        return;

    const int3 coarse_cell = grid_coordinates(coarse_index, coarse_ny, coarse_nz);
    coarse_p[coarse_index] = 0.0f;
    coarse_b[coarse_index] = 0.0f;
    coarse_mask[coarse_index] = 0.0f;

    if (!is_interior_cell(coarse_cell.x, coarse_cell.y, coarse_cell.z, coarse_nx, coarse_ny, coarse_nz))
        return;

    const float inv_delta2 = 1.0f / (fine_delta * fine_delta);
    const int fine_i = 2 * coarse_cell.x;
    const int fine_j = 2 * coarse_cell.y;
    const int fine_k = 2 * coarse_cell.z;
    float residual_sum = 0.0f;
    float residual_count = 0.0f;

    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const int i = fine_i + di;
                const int j = fine_j + dj;
                const int k = fine_k + dk;
                if (!is_interior_cell(i, j, k, fine_nx, fine_ny, fine_nz))
                    continue;
                const int fine_index = grid_index(i, j, k, fine_ny, fine_nz);
                if (fine_mask[fine_index] <= 0.0f)
                    continue;
                residual_sum += residual(fine_p, fine_b, fine_mask, inv_delta2, i, j, k, fine_nx, fine_ny, fine_nz);
                residual_count += 1.0f;
            }
        }
    }

    if (residual_count > 0.0f) {
        coarse_b[coarse_index] = residual_sum / residual_count;
        coarse_mask[coarse_index] = 1.0f;
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

    const int i = min(fine_cell.x / 2, coarse_nx - 1);
    const int j = min(fine_cell.y / 2, coarse_ny - 1);
    const int k = min(fine_cell.z / 2, coarse_nz - 1);
    fine_p[fine_index] += 0.25f * coarse_error[grid_index(i, j, k, coarse_ny, coarse_nz)];
}

__kernel void rbgs_step(__global float *p,
                        __global const float *b,
                        __global const float *mask,
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

    const float diagonal = dense_diagonal(cell.x, cell.y, cell.z, nx, ny, nz);
    p[index] = (active_neighbor_sum(p, mask, cell.x, cell.y, cell.z, ny, nz) - delta * delta * b[index]) / diagonal;
}

__kernel void coarse_smooth(__global float *p,
                            __global const float *b,
                            __global const float *mask,
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

                const float diagonal = dense_diagonal(cell.x, cell.y, cell.z, nx, ny, nz);
                p[index] =
                    (active_neighbor_sum(p, mask, cell.x, cell.y, cell.z, ny, nz) - delta2 * b[index]) / diagonal;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
