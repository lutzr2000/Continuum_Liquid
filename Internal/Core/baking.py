import bpy

from ...Solver.Kernel.solver import solver
from .export import export_config
from ..UI.node_output import output_path_valid


def get_output_node(context):
    node = getattr(context, "node", None)
    if getattr(node, "bl_idname", "") == "CONTINUUM_LIQUID_OUTPUT_NODE":
        return node
    return None


class CONTINUUM_LIQUID_OT_bake(bpy.types.Operator):
    bl_idname = "continuum_liquid.bake"
    bl_label = "Bake"
    bl_description = "Run the Continuum Liquid solver"

    def execute(self, context):
        output_node = get_output_node(context)
        if output_node is None:
            self.report({"ERROR"}, "Bake must be started from an Output node")
            return {"CANCELLED"}

        validation_error = output_path_valid(output_node.output_path)
        if validation_error is not None:
            self.report({"ERROR"}, validation_error)
            return {"CANCELLED"}

        try:
            config = export_config.build_config_dict(output_node)
            export_config.export_config_dict(config)
            solver(config)
        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, "Bake finished")
        return {"FINISHED"}
