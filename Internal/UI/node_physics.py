from bpy.props import FloatProperty

from . import node_base, sockets


class ContinuumLiquidPhysicsNode(node_base.ContinuumLiquidBaseNode):
    bl_idname = "CONTINUUM_LIQUID_PHYSICS_NODE"
    bl_label = "Physics"
    bl_icon = "MOD_PHYSICS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0
    property_groups = (
        ("Water", ("water_density", "water_viscosity")),
        ("Air", ("air_density", "air_viscosity")),
    )

    water_density: FloatProperty(
        name="Water Density",
        default=1000.0,
        min=0.0001,
        precision=4,
        options=set(),
    )  # type: ignore

    air_density: FloatProperty(
        name="Air Density",
        default=1.225,
        min=0.0001,
        precision=4,
        options=set(),
    )  # type: ignore

    water_viscosity: FloatProperty(
        name="Water Viscosity",
        default=1.0e-6,
        min=0.0,
        precision=6,
        options=set(),
    )  # type: ignore
    
    air_viscosity: FloatProperty(
        name="Air Viscosity",
        default=1.5e-5,
        min=0.0,
        precision=6,
        options=set(),
    )  # type: ignore

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumLiquidLinkSocket.bl_idname,
            "Physics",
        )

    def draw_buttons(self, _context, layout):
        for title, property_names in self.property_groups:
            self._draw_group(layout, title, property_names)
