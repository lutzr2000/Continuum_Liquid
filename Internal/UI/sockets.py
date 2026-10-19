import bpy


class _ContinuumLiquidSocket:
    socket_color = (0.90, 0.55, 0.20, 1.0)

    def draw(self, _context, layout, _node, text):
        layout.label(text=text)

    def draw_color(self, _context, _node):
        return self.socket_color


class ContinuumLiquidLinkSocket(_ContinuumLiquidSocket, bpy.types.NodeSocket):
    bl_idname = "CONTINUUM_LIQUID_LINK_SOCKET"
    bl_label = "Continuum Liquid Link"


class ContinuumLiquidForceSocket(_ContinuumLiquidSocket, bpy.types.NodeSocket):
    bl_idname = "CONTINUUM_LIQUID_FORCE_SOCKET"
    bl_label = "Continuum Liquid Force"
    socket_color = (0.45, 0.65, 0.95, 1.0)


class ContinuumLiquidReferenceFrameSocket(
    _ContinuumLiquidSocket,
    bpy.types.NodeSocket,
):
    bl_idname = "CONTINUUM_LIQUID_REFERENCE_FRAME_SOCKET"
    bl_label = "Continuum Liquid Reference Frame"
    socket_color = (1.0, 0.2, 0.65, 1.0)


class ContinuumLiquidResultSocket(_ContinuumLiquidSocket, bpy.types.NodeSocket):
    bl_idname = "CONTINUUM_LIQUID_RESULT_SOCKET"
    bl_label = "Continuum Liquid Result"
    socket_color = (0.65, 0.35, 0.85, 1.0)
