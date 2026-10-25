from bpy.props import FloatProperty

from . import node_base, sockets


class ContinuumLiquidDomainNode(node_base.ContinuumLiquidBaseNode):
    bl_idname = "CONTINUUM_LIQUID_DOMAIN_NODE"
    bl_label = "Domain"
    bl_icon = "MESH_GRID"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    resolution: FloatProperty(
        name="Resolution",
        default=0.1,
        min=0.000001,
        soft_min=0.000001,
        unit="LENGTH",
        options=set()
    )  # type: ignore
    
    lx: FloatProperty(
        name="Lx",
        default=10.0,
        min=0.000001,
        soft_min=0.000001,
        unit="LENGTH",
        options=set()
    )  # type: ignore

    ly: FloatProperty(
        name="Ly",
        default=10.0,
        min=0.000001,
        soft_min=0.000001,
        unit="LENGTH",
        options=set(),
    )  # type: ignore

    lz: FloatProperty(
        name="Lz",
        default=10.0,
        min=0.000001,
        soft_min=0.000001,
        unit="LENGTH",
        options=set(),
    )  # type: ignore

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumLiquidLinkSocket.bl_idname,
            "Domain",
        )

    def draw_buttons(self, _context, layout):
        column = layout.column(align=True)
        for property_name in ("resolution", "lx", "ly", "lz"):
            column.prop(self, property_name)
