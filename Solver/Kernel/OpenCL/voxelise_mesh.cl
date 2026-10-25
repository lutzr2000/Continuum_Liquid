inline float3 triangle_vertex(__global const float *triangles, const int triangle, const int vertex) {
    return vload3(0, triangles + triangle * 9 + vertex * 3);
}

inline int ray_hit(const float3 origin, const float3 direction, const float3 a, const float3 b, const float3 c) {
    const float3 edge1 = b - a;
    const float3 edge2 = c - a;
    const float3 p = cross(direction, edge2);
    const float determinant = dot(edge1, p);
    if (fabs(determinant) < 1.0e-7f)
        return 0;
    const float inverse = 1.0f / determinant;
    const float3 t = origin - a;
    const float u = dot(t, p) * inverse;
    if (u < 0.0f || u > 1.0f)
        return 0;
    const float3 q = cross(t, edge1);
    const float v = dot(direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f)
        return 0;
    return dot(edge2, q) * inverse > 1.0e-6f;
}

__kernel void voxelise_mesh(__global const float *triangles,
                            __global uchar *mask,
                            const int triangle_count,
                            const int nx,
                            const int ny,
                            const int nz,
                            const float delta,
                            const float ox,
                            const float oy,
                            const float oz) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    if (i >= nx || j >= ny || k >= nz || i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1)
        return;
    const float3 point =
        (float3)(ox + ((float)i + 0.5f) * delta, oy + ((float)j + 0.5f) * delta, oz + ((float)k + 0.5f) * delta);
    const float3 direction = normalize((float3)(1.0f, 0.3713907f, 0.2178311f));
    int intersections = 0;
    for (int triangle = 0; triangle < triangle_count; ++triangle)
        intersections += ray_hit(point, direction, triangle_vertex(triangles, triangle, 0),
                                 triangle_vertex(triangles, triangle, 1), triangle_vertex(triangles, triangle, 2));
    if (intersections & 1)
        mask[(i * ny + j) * nz + k] = (uchar)1;
}
