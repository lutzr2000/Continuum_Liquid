from pathlib import Path

import bpy

from .domain_grid import dimensions


def normalized_path(path_value):
    if not path_value:
        return None
    try:
        return str(Path(path_value).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


class USDResultManager:
    def __init__(self):
        self.loaded_directory = None

    def load_bake(self, filepath, simulation_node=None):
        filepath = Path(filepath).resolve()
        if not filepath.is_file():
            raise FileNotFoundError(f"USD result not found: {filepath}")

        bake_directory = filepath.parent
        self.clear_bake(bake_directory)
        objects_before = set(bpy.data.objects)
        bpy.ops.wm.usd_import(filepath=str(filepath))
        imported_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
        if not imported_objects:
            raise RuntimeError("USD import did not create an object")

        offset = self._domain_offset(simulation_node)
        imported_set = set(imported_objects)
        for obj in imported_objects:
            if getattr(obj, "parent", None) not in imported_set:
                obj.location.x += offset[0]
                obj.location.y += offset[1]
                obj.location.z += offset[2]
            obj["continuum_liquid_bake_directory"] = str(bake_directory)
            data = getattr(obj, "data", None)
            if data is not None:
                data["continuum_liquid_bake_directory"] = str(bake_directory)

        self.loaded_directory = str(bake_directory)
        return imported_objects

    def _domain_offset(self, simulation_node):
        if simulation_node is None:
            return (0.0, 0.0, 0.0)
        socket = simulation_node.inputs.get("Domain")
        if socket is None:
            return (0.0, 0.0, 0.0)
        for link in socket.links:
            domain_node = getattr(link, "from_node", None)
            if getattr(domain_node, "bl_idname", "") == "CONTINUUM_LIQUID_DOMAIN_NODE":
                width, depth, _height = dimensions(domain_node)
                return (-width * 0.5, -depth * 0.5, 0.0)
        return (0.0, 0.0, 0.0)

    def clear_bake(self, bake_directory):
        target = normalized_path(bake_directory)
        if target is None:
            return
        for obj in list(bpy.data.objects):
            if normalized_path(obj.get("continuum_liquid_bake_directory")) != target:
                continue
            data = getattr(obj, "data", None)
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and getattr(data, "users", 0) == 0:
                for collection_name in ("pointclouds", "meshes", "curves"):
                    collection = getattr(bpy.data, collection_name, None)
                    if collection is not None and data.name in collection:
                        collection.remove(data)
                        break
