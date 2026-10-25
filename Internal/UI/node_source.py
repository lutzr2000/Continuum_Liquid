from bpy.props import EnumProperty, FloatVectorProperty, IntProperty

from . import node_base, sockets


class ContinuumLiquidSourceNode(node_base.ContinuumLiquidBaseNode):
    bl_idname = "CONTINUUM_LIQUID_SOURCE_NODE"
    bl_label = "Source"
    bl_icon = "LIGHT_SUN"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_type: EnumProperty(
        name="Type",
        items=(
            ("GEOMETRY", "Geometry", "Use the geometry as initial liquid"),
            ("INFLOW", "Inflow", "Continuously emit liquid"),
            ("OUTFLOW", "Outflow", "Remove liquid inside the geometry"),
        ),
        default="GEOMETRY",
        options=set(),
    )  # type: ignore
    particle_density: IntProperty(
        name="Particle Density",
        description="Number of liquid particles generated per cell",
        default=8,
        min=1,
        max=16,
        options=set(),
    )  # type: ignore
    velocity: FloatVectorProperty(
        name="Velocity",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="VELOCITY",
        description="Initial or emitted liquid velocity",
        options={"ANIMATABLE"},
    )  # type: ignore

    def _sync_node(self):
        self._ensure_geometry_input()
        self._ensure_named_output(
            sockets.ContinuumLiquidLinkSocket.bl_idname,
            "Source",
        )

    def draw_buttons(self, _context, layout):
        column = layout.column(align=True)
        column.prop(self, "source_type", text="")
        column.prop(self, "particle_density")
        column.prop(self, "velocity")
