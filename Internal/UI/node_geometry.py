import bpy
from bpy.props import PointerProperty

from . import node_base, sockets


class ContinuumLiquidGeometryNode(node_base.ContinuumLiquidBaseNode):
    bl_idname = "CONTINUUM_LIQUID_GEOMETRY_NODE"
    bl_label = "Geometry"
    bl_icon = "OUTLINER_OB_MESH"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(name="Object", type=bpy.types.Object)  # type: ignore

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumLiquidGeometrySocket.bl_idname,
            "Geometry",
        )

    def draw_buttons(self, _context, layout):
        layout.prop(self, "source_object", text="Object")
