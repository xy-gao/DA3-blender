bl_info = {
    "name": "DA3 Addon",
    "author": "Xiangyi Gao",
    "version": (1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > DA3",
    "description": "Generate point clouds from images using DA3",
    "category": "3D View",
}

import bpy
from .dependencies import Dependencies

class DA3InstallDepsOperator(bpy.types.Operator):
    bl_idname = "da3.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Downloads and installs required dependencies. This may take a few minutes"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        try:
            Dependencies.install()
            self.report({'INFO'}, "Dependencies installed successfully. Please re-enable the addon.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to install dependencies: {e}")
        return {'FINISHED'}

class DA3AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        if not Dependencies.check():
            layout.label(text="Dependencies are not installed.")
            layout.operator(DA3InstallDepsOperator.bl_idname, text="Install Dependencies")
            layout.label(text="Please check Blender System Console for details.")
        else:
            layout.label(text="Dependencies are installed.")

def register_classes():
    from . import operators, panels
    bpy.utils.register_class(operators.DownloadModelOperator)
    bpy.utils.register_class(operators.GeneratePointCloudOperator)
    bpy.utils.register_class(panels.DA3Panel)
    bpy.types.Scene.da3_input_folder = bpy.props.StringProperty(subtype='DIR_PATH')
    bpy.types.Scene.da3_model_name = bpy.props.EnumProperty(
        items=[
            ('da3-small', 'DA3 Small', 'Small model for faster inference'),
            ('da3-base', 'DA3 Base', 'Base model with balanced performance'),
            ('da3-large', 'DA3 Large', 'Large model for better quality'),
            ('da3-giant', 'DA3 Giant', 'Giant model for highest quality'),
        ],
        name="Model",
        description="Select DA3 model variant",
        default='da3-large'
    )
    bpy.types.Scene.da3_batch_size = bpy.props.IntProperty(
        name="Batch Size",
        description="Number of images to process in a single batch",
        default=1,
        min=1,
        max=16
    )
    bpy.types.Scene.da3_progress = bpy.props.FloatProperty(
        name="Progress",
        subtype='PERCENTAGE',
        default=0.0,
        min=0.0,
        max=100.0
    )

def unregister_classes():
    from . import operators, panels
    bpy.utils.unregister_class(operators.DownloadModelOperator)
    bpy.utils.unregister_class(operators.GeneratePointCloudOperator)
    bpy.utils.unregister_class(panels.DA3Panel)
    del bpy.types.Scene.da3_input_folder
    del bpy.types.Scene.da3_model_name

classes = [
    DA3AddonPreferences,
    DA3InstallDepsOperator,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    if Dependencies.check():
        register_classes()

def unregister():
    if Dependencies.check():
        unregister_classes()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()