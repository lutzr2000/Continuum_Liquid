import bpy
from nodeitems_utils import register_node_categories, unregister_node_categories

from .Core.baking import CONTINUUM_LIQUID_OT_bake
from .UI.node_output import ContinuumLiquidOutputNode
from .UI.node_domain import ContinuumLiquidDomainNode
from .UI.node_physics import ContinuumLiquidPhysicsNode
from .UI.node_simulation import ContinuumLiquidSimulationNode
from .UI.node_tree import (
    ContinuumLiquidNodeTree,
    NODE_CATEGORIES_ID,
    build_node_categories,
)
from .UI.sockets import (
    ContinuumLiquidForceSocket,
    ContinuumLiquidLinkSocket,
    ContinuumLiquidReferenceFrameSocket,
    ContinuumLiquidResultSocket,
)

classes = (
    ContinuumLiquidNodeTree,
    ContinuumLiquidLinkSocket,
    ContinuumLiquidForceSocket,
    ContinuumLiquidReferenceFrameSocket,
    ContinuumLiquidResultSocket,
    ContinuumLiquidDomainNode,
    ContinuumLiquidPhysicsNode,
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


def reset_node_body_colors():
    for node_tree in bpy.data.node_groups:
        if node_tree.bl_idname != ContinuumLiquidNodeTree.bl_idname:
            continue
        for node in node_tree.nodes:
            node.use_custom_color = False


def register():
    for cls in classes:
        safe_register_class(cls)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    register_node_categories(NODE_CATEGORIES_ID, build_node_categories())
    reset_node_body_colors()

def unregister():
    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    for cls in reversed(classes):
        safe_unregister_class(cls)
