from datetime import datetime
from pathlib import Path

import bpy


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
        }
    }


def export_config_dict(config_dict):
    output_root = Path(config_dict["simulation"]["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bake_directory = output_root / f"Bake_run_{timestamp}"
    bake_directory.mkdir(parents=False, exist_ok=False)
    config_dict["simulation"]["output_path"] = str(bake_directory.resolve())
    return bake_directory, config_dict
