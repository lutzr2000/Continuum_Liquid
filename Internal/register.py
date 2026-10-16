import bpy
from nodeitems_utils import register_node_categories, unregister_node_categories

from .Core.baking import CONTINUUM_LIQUID_OT_bake
from .UI.node_output import ContinuumLiquidOutputNode
from .UI.node_simulation import ContinuumLiquidSimulationNode
from .UI.node_tree import (
    ContinuumLiquidNodeTree,
    NODE_CATEGORIES_ID,
    build_node_categories,
)
from .UI.sockets import ContinuumLiquidResultSocket

classes = (
    ContinuumLiquidNodeTree,
    ContinuumLiquidResultSocket,
    ContinuumLiquidSimulationNode,
    ContinuumLiquidOutputNode,
    CONTINUUM_LIQUID_OT_bake,
)


def safe_unregister_class(cls):
    try:
        bpy.utils.unregister_class(cls)
    except RuntimeError:
        pass


def safe_register_class(cls):
    safe_unregister_class(cls)
    bpy.utils.register_class(cls)

def register():
    for cls in classes:
        safe_register_class(cls)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    register_node_categories(NODE_CATEGORIES_ID, build_node_categories())

def unregister():
    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    for cls in reversed(classes):
        safe_unregister_class(cls)
