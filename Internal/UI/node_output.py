import os
from pathlib import Path

import bpy
from bpy.props import IntProperty, StringProperty

from . import node_base, sockets
from ..Core import baking as bake_main

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
    bl_width_min = 240.0
    bl_width_max = 420.0

    fps: IntProperty(
        name="FPS",
        default=24,
        min=1,
        soft_min=1,
        description="Output frame rate",
        options=set(),
    )  # type: ignore

    output_path: StringProperty(
        name="Path",
        default="",
        subtype="DIR_PATH",
        options=set(),
    )  # type: ignore

    last_bake_directory: StringProperty(
        default="",
        options={"HIDDEN"},
    )  # type: ignore

    def _sync_node(self):
        self._ensure_socket(
            self.inputs,
            sockets.ContinuumLiquidResultSocket.bl_idname,
            "Result",
        )

    def _sync_defaults_from_scene(self, context):
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        render = getattr(scene, "render", None) if scene is not None else None
        if render is not None and self.fps == 24:
            self.fps = int(getattr(render, "fps", self.fps))

    def init(self, context):
        self._apply_node_style()
        self._sync_node()
        self._sync_defaults_from_scene(context)

    def draw_buttons(self, _context, layout):
        layout.prop(self, "fps")
        layout.separator()
        layout.prop(self, "output_path")
        layout.separator()
        row = layout.row()
        validation_error = output_path_valid(self.output_path)
        row.enabled = validation_error is None and not bake_main.bake_running
        if (
            validation_error is None
            and bake_main.output_node_has_baked_data(self)
            and not bake_main.bake_running
        ):
            row.operator("continuum_liquid.free_bake", text="Free Bake", icon="TRASH")
        else:
            row.operator("continuum_liquid.bake", text="Bake", icon="RENDER_STILL")
        if validation_error is not None:
            layout.label(text=validation_error, icon="INFO")
        elif bake_main.bake_running:
            layout.label(text="Bake is running", icon="INFO")
