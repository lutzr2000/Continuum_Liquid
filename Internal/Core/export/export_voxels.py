from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..domain_grid import dimensions, grid_shape


def _linked_geometry_objects(node):
    socket = node.inputs.get("Geometry")
    if socket is None:
        return []
    objects = []
    for link in socket.links:
        geometry_node = getattr(link, "from_node", None)
        if getattr(geometry_node, "bl_idname", "") != "CONTINUUM_LIQUID_GEOMETRY_NODE":
            continue
        source_object = getattr(geometry_node, "source_object", None)
        if source_object is not None and getattr(source_object, "type", None) == "MESH":
            objects.append(source_object)
    return objects


def _object_bvh(source_object, depsgraph):
    object_eval = source_object.evaluated_get(depsgraph)
    mesh = None
    try:
        mesh = object_eval.to_mesh(
            preserve_all_data_layers=False,
            depsgraph=depsgraph,
        )
        if mesh is None or not mesh.polygons:
            return None, None, None
        matrix = object_eval.matrix_world
        vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        flip_winding = matrix.to_3x3().determinant() < 0.0
        polygons = [
            (
                tuple(reversed(polygon.vertices))
                if flip_winding
                else tuple(polygon.vertices)
            )
            for polygon in mesh.polygons
        ]
        coordinates = np.asarray(
            [tuple(vertex) for vertex in vertices], dtype=np.float64
        )
        return (
            BVHTree.FromPolygons(vertices, polygons, all_triangles=False),
            coordinates.min(axis=0),
            coordinates.max(axis=0),
        )
    finally:
        if mesh is not None:
            object_eval.to_mesh_clear()


def _voxelize_objects(objects, domain_node):
    shape = grid_shape(domain_node)
    size = dimensions(domain_node)
    delta = float(domain_node.resolution)
    domain_min = np.asarray((-0.5 * size[0], -0.5 * size[1], 0.0))
    mask = np.zeros(shape, dtype=np.bool_)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for source_object in objects:
        bvh, bounds_min, bounds_max = _object_bvh(source_object, depsgraph)
        if bvh is None:
            continue
        lower = np.maximum(0, np.floor((bounds_min - domain_min) / delta).astype(int))
        upper = np.minimum(
            shape, np.ceil((bounds_max - domain_min) / delta).astype(int)
        )
        lower = np.maximum(lower, 1)
        upper = np.minimum(upper, np.asarray(shape) - 1)
        ray_direction = Vector((1.0, 0.3713907, 0.2178311)).normalized()
        ray_distance = float(np.linalg.norm(bounds_max - bounds_min) * 2.0 + delta)
        for i in range(int(lower[0]), int(upper[0])):
            for j in range(int(lower[1]), int(upper[1])):
                for k in range(int(lower[2]), int(upper[2])):
                    point = domain_min + (np.asarray((i, j, k)) + 0.5) * delta
                    origin = Vector(point)
                    intersections = 0
                    remaining = ray_distance
                    while remaining > 0.0:
                        location, _normal, index, distance = bvh.ray_cast(
                            origin, ray_direction, remaining
                        )
                        if index is None:
                            break
                        intersections += 1
                        advance = float(distance) + max(delta * 1.0e-5, 1.0e-7)
                        origin += ray_direction * advance
                        remaining -= advance
                    if intersections % 2:
                        mask[i, j, k] = True
    return mask


def export_static_masks(
    config, bake_directory, domain_node, source_nodes, obstacle_nodes
):
    if domain_node is None:
        return
    simulation = config["simulation"]
    start_frame = int(simulation["settings"]["start_frame"])
    scene = bpy.context.scene
    original_frame = int(scene.frame_current)
    masks_directory = Path(bake_directory) / "masks"
    masks_directory.mkdir(parents=True, exist_ok=True)
    try:
        scene.frame_set(start_frame)
        for index, (source, entry) in enumerate(
            zip(source_nodes, simulation["sources"])
        ):
            mask = _voxelize_objects(_linked_geometry_objects(source), domain_node)
            relative_path = f"masks/source_{index:04d}.npz"
            np.savez_compressed(masks_directory / f"source_{index:04d}.npz", mask=mask)
            entry["mask_file"] = relative_path

        obstacle_mask = _voxelize_objects(
            [
                source_object
                for obstacle in obstacle_nodes
                for source_object in _linked_geometry_objects(obstacle)
            ],
            domain_node,
        )
        obstacle_path = masks_directory / "obstacles.npz"
        np.savez_compressed(obstacle_path, mask=obstacle_mask)
        simulation["obstacle_mask_file"] = "masks/obstacles.npz"
    finally:
        scene.frame_set(original_frame)
