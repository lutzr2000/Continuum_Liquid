import bpy


class ContinuumLiquidResultSocket(bpy.types.NodeSocket):
    bl_idname = "CONTINUUM_LIQUID_RESULT_SOCKET"
    bl_label = "Continuum Liquid Result"

    def draw(self, _context, layout, _node, text):
        layout.label(text=text)

    def draw_color(self, _context, _node):
        return (0.20, 0.55, 0.95, 1.0)
