from pathlib import Path

import numpy as np

from . import helper


def _cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _point_in_triangle(point, a, b, c, orientation, epsilon):
    return all(
        orientation * _cross_2d(end - start, point - start) >= -epsilon
        for start, end in ((a, b), (b, c), (c, a))
    )


def _triangulate_polygon(points, face):
    polygon = np.asarray(points[face], dtype=np.float64)
    following = np.roll(polygon, -1, axis=0)
    normal = np.array(
        [
            np.sum(
                (polygon[:, 1] - following[:, 1]) * (polygon[:, 2] + following[:, 2])
            ),
            np.sum(
                (polygon[:, 2] - following[:, 2]) * (polygon[:, 0] + following[:, 0])
            ),
            np.sum(
                (polygon[:, 0] - following[:, 0]) * (polygon[:, 1] + following[:, 1])
            ),
        ]
    )
    projected = np.delete(polygon, int(np.argmax(np.abs(normal))), axis=1)
    signed_area = np.sum(
        projected[:, 0] * np.roll(projected[:, 1], -1)
        - np.roll(projected[:, 0], -1) * projected[:, 1]
    )
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    epsilon = (
        np.finfo(np.float64).eps
        * max(float(np.ptp(projected, axis=0).max()), 1.0) ** 2
        * 32.0
    )
    remaining = list(range(len(face)))
    result = []
    while len(remaining) > 3:
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following_index = remaining[(position + 1) % len(remaining)]
            a, b, c = projected[[previous, current, following_index]]
            if orientation * _cross_2d(b - a, c - b) <= epsilon:
                continue
            others = [
                i for i in remaining if i not in (previous, current, following_index)
            ]
            if any(
                _point_in_triangle(projected[i], a, b, c, orientation, epsilon)
                for i in others
            ):
                continue
            result.append([face[previous], face[current], face[following_index]])
            del remaining[position]
            break
        else:
            raise ValueError(
                "Cannot triangulate a degenerate or self-intersecting USD face"
            )
    result.append([face[index] for index in remaining])
    return result


def load_usdc_triangles(path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    mesh = next(
        (UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)),
        None,
    )
    if mesh is None:
        raise ValueError(f"USD file contains no mesh: {path}")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    holes = set(int(index) for index in (mesh.GetHoleIndicesAttr().Get() or ()))
    triangles = []
    offset = 0
    for face_index, count in enumerate(counts):
        face = [int(index) for index in indices[offset : offset + int(count)]]
        offset += int(count)
        if face_index in holes or len(face) < 3:
            continue
        triangles.extend(
            [face] if len(face) == 3 else _triangulate_polygon(points, face)
        )
    if not triangles:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.ascontiguousarray(points[np.asarray(triangles, dtype=np.int32)])


def _world_triangles(entry, bake_path):
    triangles = load_usdc_triangles(Path(bake_path) / entry["mesh_file"])
    matrix = np.asarray(entry["matrix_world"], dtype=np.float32)
    flat = triangles.reshape(-1, 3)
    world = flat @ matrix[:3, :3].T + matrix[:3, 3]
    return np.ascontiguousarray(world.reshape(-1, 3, 3), dtype=np.float32)


def voxelise_entries(
    context, queue, kernel, entries, bake_path, grid_shape, delta, origin
):
    mask = helper.device_array(context, grid_shape, dtype=np.uint8)
    helper.fill_device(queue, mask, 0, dtype=np.uint8)
    for entry in entries or ():
        triangles = _world_triangles(entry, bake_path)
        if not triangles.size:
            continue
        kernel(
            queue,
            tuple(int(value) for value in grid_shape),
            None,
            helper.to_device(context, triangles),
            mask,
            np.int32(len(triangles)),
            np.int32(grid_shape[0]),
            np.int32(grid_shape[1]),
            np.int32(grid_shape[2]),
            np.float32(delta),
            np.float32(origin[0]),
            np.float32(origin[1]),
            np.float32(origin[2]),
        )
    return mask


def build_static_masks(context, queue, kernel, simulation, grid_shape, delta):
    size = simulation["domain"]["size"]
    origin = (-0.5 * float(size["x"]), -0.5 * float(size["y"]), 0.0)
    bake_path = simulation["output_path"]
    sources = []
    for source in simulation.get("sources") or ():
        mask = voxelise_entries(
            context,
            queue,
            kernel,
            source.get("geometry_inputs"),
            bake_path,
            grid_shape,
            delta,
            origin,
        )
        sources.append({**source, "mask_device": mask})
    obstacles = [
        entry
        for obstacle in simulation.get("obstacles") or ()
        for entry in obstacle.get("geometry_inputs") or ()
    ]
    obstacle_mask = voxelise_entries(
        context, queue, kernel, obstacles, bake_path, grid_shape, delta, origin
    )
    return sources, obstacle_mask
