from datetime import datetime
from pathlib import Path

import bpy

from . import export_geometry


def linked_node(node, socket_name, direction="input"):
    if direction == "input":
        socket = node.inputs.get(socket_name)
        linked_node_attr = "from_node"
    elif direction == "output":
        socket = node.outputs.get(socket_name)
        linked_node_attr = "to_node"
    else:
        raise ValueError("direction must be 'input' or 'output'")

    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        linked = getattr(link, linked_node_attr, None)
        if linked is not None:
            return linked
    return None


def linked_nodes(node, socket_name, direction="input"):
    socket = (node.inputs if direction == "input" else node.outputs).get(socket_name)
    linked_node_attr = "from_node" if direction == "input" else "to_node"
    if socket is None:
        return []
    return [
        linked
        for link in socket.links
        if (linked := getattr(link, linked_node_attr, None)) is not None
    ]


def require_linked_node(node, socket_name, expected_type, direction="input"):
    linked = linked_node(node, socket_name, direction)
    if linked is None or getattr(linked, "bl_idname", "") != expected_type:
        raise ValueError(f"{socket_name} node is not connected")
    return linked


def build_config_dict(output_node):
    output_path = Path(bpy.path.abspath(output_node.output_path)).expanduser().resolve()

    simulation_node = require_linked_node(
        output_node,
        "Result",
        "CONTINUUM_LIQUID_SIMULATION_NODE",
    )
    domain_node = require_linked_node(
        simulation_node,
        "Domain",
        "CONTINUUM_LIQUID_DOMAIN_NODE",
    )
    physics_node = require_linked_node(
        simulation_node,
        "Physics",
        "CONTINUUM_LIQUID_PHYSICS_NODE",
    )

    source_nodes = [
        node
        for node in linked_nodes(simulation_node, "Source")
        if getattr(node, "bl_idname", "") == "CONTINUUM_LIQUID_SOURCE_NODE"
    ]
    obstacle_nodes = [
        node
        for node in linked_nodes(simulation_node, "Obstacles")
        if getattr(node, "bl_idname", "") == "CONTINUUM_LIQUID_OBSTACLE_NODE"
    ]
    viewer_nodes = [
        node
        for node in linked_nodes(simulation_node, "Result", direction="output")
        if getattr(node, "bl_idname", "") == "CONTINUUM_LIQUID_VIEWER_NODE"
    ]

    start_frame = int(simulation_node.start_frame)
    scene = bpy.context.scene
    original_frame = int(scene.frame_current)
    geometry_entries = {}
    node_geometry = {}
    try:
        scene.frame_set(start_frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for node in source_nodes + obstacle_nodes:
            entries = []
            for geometry_node in linked_nodes(node, "Geometry"):
                if (
                    getattr(geometry_node, "bl_idname", "")
                    != "CONTINUUM_LIQUID_GEOMETRY_NODE"
                ):
                    continue
                source_object = getattr(geometry_node, "source_object", None)
                if (
                    source_object is None
                    or getattr(source_object, "type", None) != "MESH"
                ):
                    continue
                matrix = source_object.evaluated_get(depsgraph).matrix_world
                entries.append(
                    {
                        "object_name": source_object.name,
                        "mesh_file": f"geometry/{source_object.name}.usdc",
                        "mesh_format": "usdc",
                        "matrix_world": [
                            [float(value) for value in row] for row in matrix
                        ],
                    }
                )
                geometry_entries[source_object.name] = source_object
            node_geometry[node.name] = entries
    finally:
        scene.frame_set(original_frame)

    return {
        "simulation": {
            "node_tree": output_node.id_data.name,
            "output_path": str(output_path),
            "settings": {
                "solver_backend": str(simulation_node.solver_backend),
                "compute_devices": simulation_node.selected_opencl_devices(),
                "start_frame": start_frame,
                "end_frame": int(simulation_node.end_frame),
                "fps": int(output_node.fps),
                "substeps": int(simulation_node.substeps),
                "iterations": int(simulation_node.iterations),
                "debug_timing": any(
                    bool(getattr(node, "debug_timing", False)) for node in viewer_nodes
                ),
            },
            "domain": {
                "resolution": float(domain_node.resolution),
                "size": {
                    "x": float(domain_node.lx),
                    "y": float(domain_node.ly),
                    "z": float(domain_node.lz),
                },
            },
            "physics": {
                "water": {
                    "density": float(physics_node.water_density),
                    "viscosity": float(physics_node.water_viscosity),
                },
                "air": {
                    "density": float(physics_node.air_density),
                    "viscosity": float(physics_node.air_viscosity),
                },
            },
            "sources": [
                {
                    "type": str(node.source_type),
                    "particle_density": int(node.particle_density),
                    "velocity": [float(value) for value in node.velocity],
                    "geometry_inputs": node_geometry.get(node.name, []),
                }
                for node in source_nodes
            ],
            "obstacles": [
                {"geometry_inputs": node_geometry.get(node.name, [])}
                for node in obstacle_nodes
            ],
        },
        "_geometry_export": {
            "objects": geometry_entries,
            "start_frame": start_frame,
        },
    }


def export_config_dict(config_dict):
    output_root = Path(config_dict["simulation"]["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bake_directory = output_root / f"Bake_run_{timestamp}"
    bake_directory.mkdir(parents=False, exist_ok=False)
    config_dict["simulation"]["output_path"] = str(bake_directory.resolve())
    export_context = config_dict.pop("_geometry_export", {})
    for source_object in export_context.get("objects", {}).values():
        export_geometry.export_object_as_usdc(
            source_object,
            bake_directory / "geometry",
            export_context.get("start_frame", 1),
        )
    return bake_directory, config_dict
