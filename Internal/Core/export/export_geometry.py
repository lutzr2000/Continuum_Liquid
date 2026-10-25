from pathlib import Path

import bpy


def export_object_as_usdc(source_object, target_dir, frame):
    """Export one evaluated Blender mesh at the bake start frame."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{source_object.name}.usdc"
    scene = bpy.context.scene
    current_frame = int(scene.frame_current)
    active_object = bpy.context.view_layer.objects.active
    selected_objects = list(bpy.context.selected_objects)
    try:
        scene.frame_set(int(frame))
        bpy.ops.object.select_all(action="DESELECT")
        source_object.select_set(True)
        bpy.context.view_layer.objects.active = source_object
        bpy.ops.wm.usd_export(
            filepath=str(file_path),
            selected_objects_only=True,
            export_animation=False,
            export_materials=False,
            export_uvmaps=False,
            export_mesh_colors=False,
            export_normals=False,
            export_hair=False,
            export_armatures=False,
            export_shapekeys=False,
            export_custom_properties=False,
            export_lights=False,
            export_cameras=False,
            export_curves=False,
            export_points=False,
            export_volumes=False,
            triangulate_meshes=False,
            evaluation_mode="RENDER",
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for selected_object in selected_objects:
            if selected_object.name in bpy.data.objects:
                selected_object.select_set(True)
        bpy.context.view_layer.objects.active = active_object
        scene.frame_set(current_frame)
    return file_path
