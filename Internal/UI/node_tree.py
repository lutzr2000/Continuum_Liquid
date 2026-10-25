import bpy
from nodeitems_utils import NodeCategory, NodeItem


NODE_TREE_ID = "CONTINUUM_LIQUID_NODE_TREE"
NODE_CATEGORIES_ID = "CONTINUUM_LIQUID_NODE_CATEGORIES"


class ContinuumLiquidNodeTree(bpy.types.NodeTree):
    bl_idname = NODE_TREE_ID
    bl_label = "Continuum Liquid"
    bl_icon = "MOD_FLUIDSIM"


class ContinuumLiquidNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        space_data = getattr(context, "space_data", None)
        return (
            space_data is not None
            and getattr(space_data, "tree_type", "") == NODE_TREE_ID
        )


def build_node_categories():
    return [
        ContinuumLiquidNodeCategory(
            "CONTINUUM_LIQUID_NODES",
            "Continuum Liquid",
            items=[
                NodeItem("CONTINUUM_LIQUID_DOMAIN_NODE"),
                NodeItem("CONTINUUM_LIQUID_GEOMETRY_NODE"),
                NodeItem("CONTINUUM_LIQUID_OUTPUT_NODE"),
                NodeItem("CONTINUUM_LIQUID_PHYSICS_NODE"),
                NodeItem("CONTINUUM_LIQUID_SIMULATION_NODE"),
                NodeItem("CONTINUUM_LIQUID_SOURCE_NODE"),
                NodeItem("CONTINUUM_LIQUID_OBSTACLE_NODE"),
                NodeItem("CONTINUUM_LIQUID_VIEWER_NODE"),
            ],
        )
    ]
