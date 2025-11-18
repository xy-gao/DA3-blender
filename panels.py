import bpy
from .operators import MODEL_PATH
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
        row = layout.row()
        if os.path.exists(MODEL_PATH):
            row.label(text="Model already downloaded")
        else:
            row.operator("da3.download_model")
        layout.prop(scene, "da3_input_folder", text="Input Folder")
        row = layout.row()
        row.operator("da3.generate_point_cloud")