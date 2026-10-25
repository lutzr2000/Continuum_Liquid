from datetime import datetime
from pathlib import Path

import bpy

from . import export_voxels


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
    output_path = Path(
        bpy.path.abspath(output_node.output_path)
    ).expanduser().resolve()

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

    return {
        "simulation": {
            "node_tree": output_node.id_data.name,
            "output_path": str(output_path),
            "settings": {
                "solver_backend": str(simulation_node.solver_backend),
                "start_frame": int(simulation_node.start_frame),
                "end_frame": int(simulation_node.end_frame),
                "fps": int(output_node.fps),
                "substeps": int(simulation_node.substeps),
                "iterations": int(simulation_node.iterations),
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
                }
                for node in source_nodes
            ],
            "obstacles": [{} for _node in obstacle_nodes],
        },
        "_voxel_export": {
            "domain_node": domain_node,
            "source_nodes": source_nodes,
            "obstacle_nodes": obstacle_nodes,
        },
    }


def export_config_dict(config_dict):
    output_root = Path(config_dict["simulation"]["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bake_directory = output_root / f"Bake_run_{timestamp}"
    bake_directory.mkdir(parents=False, exist_ok=False)
    config_dict["simulation"]["output_path"] = str(bake_directory.resolve())
    export_context = config_dict.pop("_voxel_export", {})
    export_voxels.export_static_masks(
        config_dict,
        bake_directory,
        export_context.get("domain_node"),
        export_context.get("source_nodes", ()),
        export_context.get("obstacle_nodes", ()),
    )
    return bake_directory, config_dict
