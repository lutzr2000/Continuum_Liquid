import json
from pathlib import Path

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

from .domain_grid import dimensions

CACHE_FILENAME = "particle_cache.json"
FLUID_COLOR = (0.04, 0.32, 1.0, 1.0)


def normalized_path(path_value):
    if not path_value:
        return None
    try:
        return str(Path(path_value).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


class ParticleCacheManager:
    def __init__(self):
        self.loaded_directory = None
        self.metadata = None
        self.positions = None
        self.loaded_frame = None
        self.draw_handler = None
        self.batch = None
        self.batch_frame = None
        self.point_size = 4.0
        self.color = FLUID_COLOR
        self.offset = (0.0, 0.0, 0.0)

    def load_bake(self, directory, simulation_node=None):
        directory = Path(directory).resolve()
        metadata_path = directory / CACHE_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Particle cache not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dtype") != "float32" or metadata.get("components") != 3:
            raise ValueError("Unsupported particle cache format")

        self.loaded_directory = str(directory)
        self.metadata = metadata
        self.offset = self._domain_offset(simulation_node)
        self.loaded_frame = None
        self.positions = None
        self.enable()
        scene = getattr(bpy.context, "scene", None)
        self.update_frame(getattr(scene, "frame_current", 1))
        return self

    def enable(self):
        if self.draw_handler is None:
            self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                self.draw, (), "WINDOW", "POST_VIEW"
            )
        self.redraw()

    def disable(self):
        if self.draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, "WINDOW")
            self.draw_handler = None
        self.positions = None
        self.loaded_frame = None
        self.batch = None
        self.batch_frame = None
        self.redraw()

    def clear_bake(self, bake_directory):
        if normalized_path(bake_directory) == normalized_path(self.loaded_directory):
            self.disable()
            self.loaded_directory = None
            self.metadata = None

    def update_frame(self, frame):
        if self.metadata is None or self.loaded_directory is None:
            return
        start = int(self.metadata["start_frame"])
        end = int(self.metadata["end_frame"])
        frame = min(end, max(start, int(frame)))
        if frame == self.loaded_frame:
            return
        frame_directory = Path(self.loaded_directory) / self.metadata["frame_directory"]
        filename = self.metadata["frame_pattern"].format(frame=frame)
        filepath = frame_directory / filename
        if not filepath.is_file():
            available = [
                candidate
                for candidate in range(start, end + 1)
                if (
                    frame_directory
                    / self.metadata["frame_pattern"].format(frame=candidate)
                ).is_file()
            ]
            if not available:
                return
            frame = min(available, key=lambda candidate: abs(candidate - frame))
            filepath = frame_directory / self.metadata["frame_pattern"].format(
                frame=frame
            )
        values = np.fromfile(filepath, dtype=np.float32)
        expected = int(self.metadata["particle_count"]) * 3
        if values.size != expected:
            raise ValueError(f"Invalid particle frame size: {filepath}")
        positions = values.reshape((-1, 3))
        positions += np.asarray(self.offset, dtype=np.float32)
        self.positions = positions
        self.loaded_frame = frame
        self.batch = None
        self.batch_frame = None
        self.redraw()

    def draw(self):
        if self.positions is None or not self._overlays_enabled():
            return
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.point_size_set(float(self.point_size))
        shader.bind()
        shader.uniform_float("color", self.color)
        if self.batch is None or self.batch_frame != self.loaded_frame:
            self.batch = batch_for_shader(shader, "POINTS", {"pos": self.positions})
            self.batch_frame = self.loaded_frame
        self.batch.draw(shader)
        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")

    @staticmethod
    def _overlays_enabled():
        space = getattr(bpy.context, "space_data", None)
        overlay = getattr(space, "overlay", None)
        return overlay is None or bool(getattr(overlay, "show_overlays", True))

    @staticmethod
    def _domain_offset(simulation_node):
        if simulation_node is not None:
            socket = simulation_node.inputs.get("Domain")
            if socket is not None:
                for link in socket.links:
                    domain = getattr(link, "from_node", None)
                    if getattr(domain, "bl_idname", "") == "CONTINUUM_LIQUID_DOMAIN_NODE":
                        width, depth, _height = dimensions(domain)
                        return (-width * 0.5, -depth * 0.5, 0.0)
        return (0.0, 0.0, 0.0)

    @staticmethod
    def redraw():
        manager = getattr(bpy.context, "window_manager", None)
        if manager is None:
            return
        for window in manager.windows:
            screen = getattr(window, "screen", None)
            for area in getattr(screen, "areas", ()):
                if area.type == "VIEW_3D":
                    area.tag_redraw()


PARTICLE_CACHE = ParticleCacheManager()


@persistent
def particle_cache_frame_change(scene, _depsgraph=None):
    PARTICLE_CACHE.update_frame(scene.frame_current)
