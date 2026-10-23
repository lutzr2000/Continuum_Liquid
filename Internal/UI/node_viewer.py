from . import node_base, sockets
from ..Core import viewer


class ContinuumLiquidViewerNode(node_base.ContinuumLiquidBaseNode):
    """Node used as a lightweight endpoint for inspecting simulation results."""

    bl_idname = "CONTINUUM_LIQUID_VIEWER_NODE"
    bl_label = "Viewer"
    bl_icon = "HIDE_OFF"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 520.0

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
