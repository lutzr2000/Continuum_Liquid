import bpy
import gpu
from . import baking
from gpu_extras.batch import batch_for_shader

from .domain_grid import dimensions
from . import load_result

draw_handler = None
current_drawn_domain = None


class ContinuumLiquid_OT_viewer_toggle_domain(bpy.types.Operator):
    """Toggle the linked domain preview in the 3D viewport."""

    bl_idname = "continuum_liquid.viewer_toggle_domain"
    bl_label = "Toggle Domain Preview"
    bl_description = "Toggle the linked Continuum Liquid domain viewport preview"

    def execute(self, context):
        node = getattr(context, "node", None)
        domain_node = get_linked_domain_node(node)

        node_tree = getattr(domain_node, "id_data", None)
        if domain_preview_enabled(domain_node):
            disable_domain_preview()
            self.report({"INFO"}, "Domain preview disabled.")
        else:
            enable_domain_preview(domain_node, node_tree)
            self.report({"INFO"}, "Domain preview enabled.")
        return {"FINISHED"}


class ContinuumLiquid_OT_viewer_toggle_particles(bpy.types.Operator):
    """Toggle the baked particle cache in the 3D viewport."""

    bl_idname = "continuum_liquid.viewer_toggle_particles"
    bl_label = "Toggle Particles"
    bl_description = "Show or hide the baked liquid particles"

    def execute(self, context):
        node = getattr(context, "node", None)
        manager = load_result.PARTICLE_CACHE
        if manager.draw_handler is not None:
            manager.disable()
            return {"FINISHED"}

        simulation_node = get_linked_simulation_node(node)
        if simulation_node is None:
            self.report({"ERROR"}, "Connect Simulation -> Viewer first")
            return {"CANCELLED"}
        output_node = get_linked_output_node(simulation_node)
        if output_node is None:
            self.report({"ERROR"}, "No Output node is connected to this simulation")
            return {"CANCELLED"}

        directory = baking.get_bake_directory(output_node)
        if directory is None:
            self.report({"ERROR"}, "No particle bake found")
            return {"CANCELLED"}
        manager.point_size = float(getattr(node, "particle_point_size", 4.0))
        manager.color = tuple(getattr(node, "particle_color", (0.04, 0.32, 1.0, 1.0)))

        manager.load_bake(directory, simulation_node=simulation_node)

        return {"FINISHED"}


def get_linked_domain_node(node):
    """Resolve Domain -> Simulation by traversing the viewer's Result input."""
    if node is None:
        return None

    result_socket = node.inputs.get("Result")
    if result_socket is None:
        return None

    for link in result_socket.links:
        simulation_node = getattr(link, "from_node", None)
        if (
            getattr(simulation_node, "bl_idname", "")
            != "CONTINUUM_LIQUID_SIMULATION_NODE"
        ):
            continue

        domain_socket = simulation_node.inputs.get("Domain")
        if domain_socket is None:
            continue

        for domain_link in domain_socket.links:
            domain_node = getattr(domain_link, "from_node", None)
            if getattr(domain_node, "bl_idname", "") == "CONTINUUM_LIQUID_DOMAIN_NODE":
                return domain_node

    return None


def get_linked_simulation_node(node):
    if node is None:
        return None
    socket = node.inputs.get("Result")

    if socket is not None:
        for link in socket.links:
            simulation = getattr(link, "from_node", None)
            if (
                getattr(simulation, "bl_idname", "")
                == "CONTINUUM_LIQUID_SIMULATION_NODE"
            ):
                return simulation

    return None


def get_linked_output_node(simulation_node):
    node_tree = getattr(simulation_node, "id_data", None)
    if node_tree is None:
        return None

    for node in node_tree.nodes:
        if getattr(node, "bl_idname", "") != "CONTINUUM_LIQUID_OUTPUT_NODE":
            continue

        socket = node.inputs.get("Result")
        if socket is not None and any(
            getattr(link, "from_node", None) == simulation_node for link in socket.links
        ):
            return node

    return None


def enable_domain_preview(domain_node, node_tree):
    """Enable the preview for the given domain node."""
    global draw_handler, current_drawn_domain

    current_drawn_domain = {
        "node_tree_name": str(getattr(node_tree, "name", "")),
        "node_name": str(getattr(domain_node, "name", "")),
    }

    if draw_handler is None:
        draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_preview, (), "WINDOW", "POST_VIEW"
        )

    redraw_viewport()
    redraw_ui()


def disable_domain_preview():
    """Disable any active domain preview."""
    global draw_handler, current_drawn_domain

    current_drawn_domain = None

    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        draw_handler = None

    redraw_viewport()
    redraw_ui()


def redraw_viewport():
    """Request a redraw for all visible 3D viewports."""
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue

        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def redraw_ui():
    """Request a redraw for node editors and property areas."""
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue

        for area in screen.areas:
            if area.type in {"NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


def draw_preview():
    """Draw the active domain preview in the 3D viewport."""
    if not current_drawn_domain:
        return

    node_tree = bpy.data.node_groups.get(current_drawn_domain.get("node_tree_name", ""))
    if node_tree is None:
        disable_domain_preview()
        return

    active_domain = node_tree.nodes.get(current_drawn_domain.get("node_name", ""))

    if not overlay_enabled():
        return

    try:
        domain_lines, cell_lines = build_segments(active_domain)
    except Exception:
        disable_domain_preview()
        return

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(2.0)

    shader.bind()
    shader.uniform_float("color", (1.0, 0.65, 0.20, 1.0))
    batch_for_shader(shader, "LINES", {"pos": domain_lines}).draw(shader)

    shader.uniform_float("color", (0.20, 0.85, 1.0, 1.0))
    batch_for_shader(shader, "LINES", {"pos": cell_lines}).draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


def overlay_enabled():
    """Return whether overlays are enabled in the currently drawing 3D view."""
    space_data = getattr(bpy.context, "space_data", None)

    if space_data is None or getattr(space_data, "type", "") != "VIEW_3D":
        return True

    overlay = getattr(space_data, "overlay", None)

    if overlay is None:
        return True

    return bool(getattr(overlay, "show_overlays", True))


def build_segments(domain_node):
    """Build line data for the full domain and one sample cell."""
    resolution = float(domain_node.resolution)
    width, depth, height = dimensions(domain_node)

    min_corner = (-(width * 0.5), -(depth * 0.5), 0.0)
    max_corner = (
        min_corner[0] + width,
        min_corner[1] + depth,
        min_corner[2] + height,
    )

    domain_lines = box_segments(min_corner, max_corner)

    cell_lines = box_segments(
        min_corner,
        (
            min_corner[0] + min(resolution, width),
            min_corner[1] + min(resolution, depth),
            min_corner[2] + min(resolution, height),
        ),
    )

    return domain_lines, cell_lines


def box_segments(min_corner, max_corner):
    """Build the line segments for a wireframe box."""
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner

    corners = (
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    )

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    segments = []
    for start_index, end_index in edges:
        segments.extend((corners[start_index], corners[end_index]))

    return segments


def domain_preview_enabled(domain_node):
    """Return whether the preview is active for the given domain node."""
    if not current_drawn_domain or domain_node is None:
        return False
    
    node_tree = bpy.data.node_groups.get(current_drawn_domain.get("node_tree_name", ""))
    if node_tree is None:
        return False
    
    active_domain = node_tree.nodes.get(current_drawn_domain.get("node_name", ""))
    if active_domain is None:
        return False

    active_tree = getattr(active_domain, "id_data", None)
    target_tree = getattr(domain_node, "id_data", None)
    return (
        active_tree is not None
        and target_tree is not None
        and active_tree.name == target_tree.name
        and active_domain.name == domain_node.name
    )


def viewer_domain_preview_enabled(viewer_node):
    """Return whether this viewer controls the active domain preview."""
    return domain_preview_enabled(get_linked_domain_node(viewer_node))


classes = (
    ContinuumLiquid_OT_viewer_toggle_domain,
    ContinuumLiquid_OT_viewer_toggle_particles,
)
