#ifndef CONTINUUM_LIQUID_HELPER_CL
#define CONTINUUM_LIQUID_HELPER_CL

inline int grid_index(const int i, const int j, const int k, const int ny, const int nz) {
    return (i * ny + j) * nz + k;
}

inline int3 grid_coordinates(const int index, const int ny, const int nz) {
    const int yz = ny * nz;
    const int i = index / yz;
    const int remainder = index - i * yz;
    const int j = remainder / nz;
    const int k = remainder - j * nz;
    return (int3)(i, j, k);
}

inline int is_inside_grid(const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz;
}

inline int is_interior_cell(const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    return i > 0 && j > 0 && k > 0 && i < nx - 1 && j < ny - 1 && k < nz - 1;
}

inline float3 gradient(__global const float *field,
                       const int i,
                       const int j,
                       const int k,
                       const float inv_dx,
                       const int ny,
                       const int nz) {
    const float center = field[grid_index(i, j, k, ny, nz)];
    const float dx = field[grid_index(i + 1, j, k, ny, nz)] - center;
    const float dy = field[grid_index(i, j + 1, k, ny, nz)] - center;
    const float dz = field[grid_index(i, j, k + 1, ny, nz)] - center;
    return (float3)(dx, dy, dz) * inv_dx;
}

inline float divergence(__global const float *u,
                        __global const float *v,
                        __global const float *w,
                        const int i,
                        const int j,
                        const int k,
                        const float inv_dx,
                        const int ny,
                        const int nz) {
    const float du = u[grid_index(i, j, k, ny, nz)] - u[grid_index(i - 1, j, k, ny, nz)];
    const float dv = v[grid_index(i, j, k, ny, nz)] - v[grid_index(i, j - 1, k, ny, nz)];
    const float dw = w[grid_index(i, j, k, ny, nz)] - w[grid_index(i, j, k - 1, ny, nz)];
    return (du + dv + dw) * inv_dx;
}

inline float laplacian(__global const float *field,
                       const int i,
                       const int j,
                       const int k,
                       const float inv_dx2,
                       const int ny,
                       const int nz) {
    const float center = field[grid_index(i, j, k, ny, nz)];
    const float neighbor_sum = field[grid_index(i + 1, j, k, ny, nz)] + field[grid_index(i - 1, j, k, ny, nz)] +
                               field[grid_index(i, j + 1, k, ny, nz)] + field[grid_index(i, j - 1, k, ny, nz)] +
                               field[grid_index(i, j, k + 1, ny, nz)] + field[grid_index(i, j, k - 1, ny, nz)];
    return (neighbor_sum - 6.0f * center) * inv_dx2;
}

#endif
