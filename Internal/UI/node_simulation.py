from . import node_base, sockets


class ContinuumLiquidSimulationNode(node_base.ContinuumLiquidBaseNode):
    bl_idname = "CONTINUUM_LIQUID_SIMULATION_NODE"
    bl_label = "Simulation"
    bl_icon = "TIME"
    bl_width_default = 260.0

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumLiquidResultSocket.bl_idname,
            "Result",
        )
