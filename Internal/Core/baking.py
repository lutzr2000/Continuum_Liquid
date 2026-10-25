import shutil
import threading
import traceback
from pathlib import Path

import bpy

from ...Solver.General import main as solver_main
from . import load_result
from .export import export_config

PARTICLE_RESULTS = load_result.PARTICLE_CACHE
bake_running = False
progress = 0.0
progress_text = "0%"
particle_count = 0
particle_stats_available = False
status_workspace = None


def get_output_node(context):
    node = getattr(context, "node", None)
    if getattr(node, "bl_idname", "") == "CONTINUUM_LIQUID_OUTPUT_NODE":
        return node

    return None


def get_connected_simulation_node(output_node):
    if output_node is None:
        return None

    result_socket = output_node.inputs.get("Result")
    if result_socket is None:
        return None

    for link in result_socket.links:
        node = getattr(link, "from_node", None)
        if getattr(node, "bl_idname", "") == "CONTINUUM_LIQUID_SIMULATION_NODE":
            return node

    return None


def ui_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue

        for area in screen.areas:
            if area.type in {
                "STATUSBAR",
                "NODE_EDITOR",
                "PROPERTIES",
                "TIMELINE",
                "DOPESHEET_EDITOR",
                "VIEW_3D",
            }:
                area.tag_redraw()


def draw_bake_progress(self, _context):
    layout = self.layout
    layout.separator_spacer()
    layout.label(text="Bake:")
    progress_row = layout.row(align=True)
    progress_row.ui_units_x = 13
    progress_row.progress(
        factor=float(progress),
        type="BAR",
        text=progress_text,
    )
    layout.separator_spacer()


def set_status_progress(context):
    global status_workspace
    workspace = getattr(context, "workspace", None)
    if workspace is None:
        return

    status_workspace = workspace
    workspace.status_text_set(draw_bake_progress)
    ui_redraw()


def set_bake_progress(current_frames, total_frames):
    global progress, progress_text
    total_frames = max(1, int(total_frames))
    progress = min(1.0, max(0.0, float(current_frames) / total_frames))
    progress_text = f"{int(round(progress * 100.0))}%"
    ui_redraw()


def clear_status_progress(context):
    global status_workspace, progress, progress_text
    workspace = status_workspace or getattr(context, "workspace", None)
    if workspace is not None:
        workspace.status_text_set(None)

    status_workspace = None
    progress = 0.0
    progress_text = "0%"
    ui_redraw()


class CONTINUUM_LIQUID_OT_bake(bpy.types.Operator):
    bl_idname = "continuum_liquid.bake"
    bl_label = "Bake"
    bl_description = "Run the solver and write a new particle cache"

    def execute(self, context):
        global bake_running, particle_count, particle_stats_available

        self.output_node = get_output_node(context)
        self.simulation_node = get_connected_simulation_node(self.output_node)

        config = export_config.build_config_dict(self.output_node)
        self.bake_directory, config = export_config.export_config_dict(config)

        self.job_result = None
        self.event_timer = None
        self.cancel_requested = False
        self.cancel_event = threading.Event()
        config.setdefault("meta", {})["cancel_event"] = self.cancel_event
        settings = config["simulation"].get("settings") or {}
        self.start_frame = int(settings.get("start_frame", 1))
        self.total_frames = max(1, int(settings.get("end_frame", 1)) - self.start_frame)
        self.completed_frames = 0
        particle_count = 0
        particle_stats_available = False
        bake_running = True
        scene = getattr(context, "scene", None)

        if scene is not None and scene.frame_current != self.start_frame:
            scene.frame_set(self.start_frame)

        set_bake_progress(0, self.total_frames)
        set_status_progress(context)
        ui_redraw()

        self.worker = threading.Thread(
            target=self._run_solver,
            args=(config,),
            name="ContinuumLiquidSolver",
            daemon=True,
        )

        self.worker.start()
        self.event_timer = context.window_manager.event_timer_add(
            0.1, window=context.window
        )

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _run_solver(self, config):
        try:
            solver_main.set_message_callback(self._handle_solver_message)
            solver_main.main(config)
            self.job_result = {"success": True}
        except Exception as exc:
            self.job_result = {
                "success": False,
                "message": str(exc) or "Bake failed",
                "traceback": traceback.format_exc(),
            }
        finally:
            solver_main.clear_message_callback()

    def _handle_solver_message(self, message):
        global particle_count, particle_stats_available
        if not isinstance(message, dict) or message.get("type") != "stats":
            return
        self.completed_frames = int(message.get("frame", 0))
        self.total_frames = max(1, int(message.get("total_frames", self.total_frames)))
        particle_count = max(0, int(message.get("particle_count", particle_count)))
        particle_stats_available = "particle_count" in message

    def modal(self, context, event):
        if event.type == "ESC":
            if not self.cancel_requested:
                self.cancel_requested = True
                self.cancel_event.set()
                self.report({"INFO"}, "Bake cancellation requested")
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        set_bake_progress(self.completed_frames, self.total_frames)
        scene = getattr(context, "scene", None)
        blender_frame = self.start_frame + min(
            self.completed_frames,
            self.total_frames,
        )

        if scene is not None and scene.frame_current != blender_frame:
            scene.frame_set(blender_frame)
        if self.job_result is None:
            return {"PASS_THROUGH"}

        global bake_running
        bake_running = False

        if self.event_timer is not None:
            context.window_manager.event_timer_remove(self.event_timer)
            self.event_timer = None

        result = self.job_result
        self.job_result = None

        clear_status_progress(context)
        ui_redraw()

        if not result["success"]:
            print(result.get("traceback", ""))
            try:
                shutil.rmtree(self.bake_directory)
            except OSError:
                pass
            self.report({"ERROR"}, result.get("message", "Bake failed"))
            return {"CANCELLED"}

        PARTICLE_RESULTS.load_bake(
            self.bake_directory, simulation_node=self.simulation_node
        )

        self.output_node.last_bake_directory = str(self.bake_directory)

        if self.cancel_requested:
            self.report({"INFO"}, "Bake cancelled; partial result loaded")
            return {"CANCELLED"}

        self.report({"INFO"}, "Bake finished")
        return {"FINISHED"}


def normalize_directory_path(path_value):
    if not path_value:
        return None
    try:
        return Path(path_value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def directory_has_cache(directory):
    directory = normalize_directory_path(directory)
    return bool(
        directory
        and directory.is_dir()
        and (directory / "particle_cache.json").is_file()
    )


def get_latest_bake_directory(output_node):
    root = normalize_directory_path(bpy.path.abspath(output_node.output_path))
    if root is None or not root.is_dir():
        return None

    candidates = [child for child in root.iterdir() if directory_has_cache(child)]

    if directory_has_cache(root):
        candidates.append(root)

    return (
        max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    )


def get_bake_directory(output_node):
    stored = normalize_directory_path(getattr(output_node, "last_bake_directory", ""))

    if directory_has_cache(stored):
        return stored

    return get_latest_bake_directory(output_node)


def output_node_has_baked_data(output_node):
    return get_bake_directory(output_node) is not None


class CONTINUUM_LIQUID_OT_free_bake(bpy.types.Operator):
    bl_idname = "continuum_liquid.free_bake"
    bl_label = "Free Bake"
    bl_description = "Remove the loaded result and delete its particle cache"

    def execute(self, context):
        output_node = get_output_node(context)
        bake_directory = get_bake_directory(output_node)
        if bake_directory is None:
            self.report({"INFO"}, "No particle bake found to remove")
            return {"FINISHED"}

        PARTICLE_RESULTS.clear_bake(bake_directory)

        root = normalize_directory_path(bpy.path.abspath(output_node.output_path))
        if bake_directory == root:
            shutil.rmtree(bake_directory / "particles", ignore_errors=True)
            (bake_directory / "particle_cache.json").unlink(missing_ok=True)
        else:
            shutil.rmtree(bake_directory)

        output_node.last_bake_directory = ""
        ui_redraw()
        self.report({"INFO"}, "Bake removed")
        return {"FINISHED"}


classes = (
    CONTINUUM_LIQUID_OT_free_bake,
    CONTINUUM_LIQUID_OT_bake,
)
