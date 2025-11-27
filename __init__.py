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

def unregister_classes():
    from . import operators, panels
    bpy.utils.unregister_class(operators.DownloadModelOperator)
    bpy.utils.unregister_class(operators.GeneratePointCloudOperator)
    bpy.utils.unregister_class(panels.DA3Panel)
    del bpy.types.Scene.da3_input_folder
    del bpy.types.Scene.da3_model_name

class DA3InstallDepsOperator(bpy.types.Operator):
    bl_idname = "da3.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Downloads and installs required dependencies for the DA3 addon. This may take a few minutes"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        try:
            Dependencies.install()
            self.report({'INFO'}, "DA3 dependencies installed successfully. Please re-enable the addon.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to install DA3 dependencies: {e}")
        return {'FINISHED'}

class DA3InstallDepsPanel(bpy.types.Panel):
    bl_label = "DA3 Dependencies"
    bl_idname = "VIEW3D_PT_da3_deps"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DA3"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Dependencies are not installed.")
        layout.operator(DA3InstallDepsOperator.bl_idname)
        layout.label(text="Please check Blender System Console for details.")

def register():
    if Dependencies.check():
        register_classes()
    else:
        bpy.utils.register_class(DA3InstallDepsOperator)
        bpy.utils.register_class(DA3InstallDepsPanel)

def unregister():
    if Dependencies.check():
        unregister_classes()
    else:
        bpy.utils.unregister_class(DA3InstallDepsOperator)
        bpy.utils.unregister_class(DA3InstallDepsPanel)

if __name__ == "__main__":
    register()