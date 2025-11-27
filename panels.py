import bpy
from .operators import get_model_path
import os

class DA3Panel(bpy.types.Panel):
    bl_label = "DA3"
    bl_idname = "VIEW3D_PT_da3"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DA3"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Model selection dropdown
        layout.prop(scene, "da3_model_name", text="Model")
        
        # Download button or status
        model_path = get_model_path(scene.da3_model_name)
        row = layout.row()
        if os.path.exists(model_path):
            row.label(text=f"Model {scene.da3_model_name} ready")
        else:
            row.operator("da3.download_model", text=f"Download {scene.da3_model_name}")
        
        layout.prop(scene, "da3_input_folder", text="Input Folder")

        layout.prop(scene, "da3_batch_size", text="Batch Size")

        row = layout.row()
        row.operator("da3.generate_point_cloud")

        # Progress bar
        if context.scene.da3_progress > 0 and context.scene.da3_progress < 100:
            layout.progress(factor=context.scene.da3_progress / 100.0)