import os
from pathlib import Path

import bpy
from bpy.props import StringProperty

from . import node_base, sockets


INVALID_OUTPUT_PATH_MESSAGE = "Invalid output path"


def output_path_valid(path_value):
    path_text = str(path_value or "").strip()
    if not path_text or "\0" in path_text:
        return INVALID_OUTPUT_PATH_MESSAGE

    try:
        output_path = Path(bpy.path.abspath(path_text)).expanduser().resolve()
        if output_path.exists():
            if not output_path.is_dir() or not os.access(output_path, os.W_OK):
                return INVALID_OUTPUT_PATH_MESSAGE
            return None

        parent = output_path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            return INVALID_OUTPUT_PATH_MESSAGE
    except (OSError, RuntimeError, TypeError, ValueError):
        return INVALID_OUTPUT_PATH_MESSAGE

    return None


class ContinuumLiquidOutputNode(node_base.ContinuumLiquidBaseNode):
    bl_idname = "CONTINUUM_LIQUID_OUTPUT_NODE"
    bl_label = "Output"
    bl_icon = "OUTPUT"
    bl_width_default = 260.0

    output_path: StringProperty(
        name="Path",
        default="",
        subtype="DIR_PATH",
        options=set(),
    )  # type: ignore

    def _sync_node(self):
        self._ensure_socket(
            self.inputs,
            sockets.ContinuumLiquidResultSocket.bl_idname,
            "Result",
        )

    def draw_buttons(self, _context, layout):
        layout.prop(self, "output_path")
        row = layout.row()
        validation_error = output_path_valid(self.output_path)
        row.enabled = validation_error is None
        row.operator("continuum_liquid.bake", text="Bake", icon="RENDER_STILL")
        if validation_error is not None:
            layout.label(text=validation_error, icon="INFO")
