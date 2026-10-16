from pathlib import Path

import bpy


def build_config_dict(output_node):
    output_path = Path(
        bpy.path.abspath(output_node.output_path)
    ).expanduser().resolve()

    return {
        "simulation": {
            "node_tree": output_node.id_data.name,
            "output_path": str(output_path),
        }
    }


def export_config_dict(config_dict):
    output_path = Path(config_dict["simulation"]["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    return config_dict
