from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty

from . import node_base, sockets
from ..Core import baking, load_result, viewer


def update_particle_display(self, _context):
    manager = load_result.PARTICLE_CACHE
    manager.point_size = float(self.particle_point_size)
    manager.color = tuple(self.particle_color)
    manager.redraw()


class ContinuumLiquidViewerNode(node_base.ContinuumLiquidBaseNode):
    """Node used as a lightweight endpoint for inspecting simulation results."""

    bl_idname = "CONTINUUM_LIQUID_VIEWER_NODE"
    bl_label = "Viewer"
    bl_icon = "HIDE_OFF"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 520.0

    particle_point_size: FloatProperty(
        name="Point Size",
        default=4.0,
        min=1.0,
        max=20.0,
        update=update_particle_display,
        options=set(),
    )  # type: ignore

    particle_color: FloatVectorProperty(
        name="Fluid Color",
        subtype="COLOR",
        size=4,
        default=(0.04, 0.32, 1.0, 1.0),
        min=0.0,
        max=1.0,
        update=update_particle_display,
        options=set(),
    )  # type: ignore

    debug_timing: BoolProperty(
        name="Debug Timing",
        description="Collect detailed solver timings (adds synchronization overhead)",
        default=False,
        options=set(),
    )  # type: ignore

    def _sync_node(self):
        self._ensure_socket(
            self.inputs,
            sockets.ContinuumLiquidResultSocket.bl_idname,
            "Result",
        )

    def free(self):
        if viewer.viewer_domain_preview_enabled(self):
            viewer.disable_domain_preview()

    def draw_buttons(self, _context, layout):
        is_domain_preview_active = viewer.viewer_domain_preview_enabled(self)
        column = layout.column(align=True)
        column.operator(
            "continuum_liquid.viewer_toggle_domain",
            text="Hide Domain" if is_domain_preview_active else "Show Domain",
            icon="HIDE_ON" if is_domain_preview_active else "HIDE_OFF",
        )
        particles_visible = load_result.PARTICLE_CACHE.draw_handler is not None
        column.separator()
        column.prop(self, "particle_point_size")
        column.prop(self, "particle_color")
        column.operator(
            "continuum_liquid.viewer_toggle_particles",
            text="Hide Particles" if particles_visible else "Show Particles",
            icon="HIDE_ON" if particles_visible else "HIDE_OFF",
        )
        if baking.particle_stats_available:
            column.separator()
            particles_text = f"{int(baking.particle_count):,}".replace(",", ".")
            column.label(text=f"Particles: {particles_text}")
        column.separator()
        column.prop(self, "debug_timing")
