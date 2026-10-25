import bpy
from nodeitems_utils import register_node_categories, unregister_node_categories

from .Core import baking, load_result, solver_status, viewer
from .UI.node_output import ContinuumLiquidOutputNode
from .UI.node_domain import ContinuumLiquidDomainNode
from .UI.node_geometry import ContinuumLiquidGeometryNode
from .UI.node_obstacle import ContinuumLiquidObstacleNode
from .UI.node_physics import ContinuumLiquidPhysicsNode
from .UI.node_simulation import ContinuumLiquidSimulationNode
from .UI.node_source import ContinuumLiquidSourceNode
from .UI.node_viewer import ContinuumLiquidViewerNode
from .UI.node_tree import (
    ContinuumLiquidNodeTree,
    NODE_CATEGORIES_ID,
    build_node_categories,
)
from .UI.sockets import (
    ContinuumLiquidForceSocket,
    ContinuumLiquidGeometrySocket,
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
    ContinuumLiquidGeometrySocket,
    ContinuumLiquidDomainNode,
    ContinuumLiquidGeometryNode,
    ContinuumLiquidPhysicsNode,
    ContinuumLiquidSimulationNode,
    ContinuumLiquidSourceNode,
    ContinuumLiquidObstacleNode,
    ContinuumLiquidViewerNode,
    ContinuumLiquidOutputNode,
    *viewer.classes,
    *baking.classes,
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
    solver_status.refresh_opencl_devices()
    for cls in classes:
        safe_register_class(cls)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    register_node_categories(NODE_CATEGORIES_ID, build_node_categories())
    if (
        load_result.particle_cache_frame_change
        not in bpy.app.handlers.frame_change_post
    ):
        bpy.app.handlers.frame_change_post.append(
            load_result.particle_cache_frame_change
        )


def unregister():
    viewer.disable_domain_preview()
    load_result.PARTICLE_CACHE.disable()
    if load_result.particle_cache_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(
            load_result.particle_cache_frame_change
        )

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    for cls in reversed(classes):
        safe_unregister_class(cls)
