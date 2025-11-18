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


def register():
    if not Dependencies.check():
        Dependencies.install()
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.register_class(operators.DownloadModelOperator)
        bpy.utils.register_class(operators.GeneratePointCloudOperator)
        bpy.utils.register_class(panels.DA3Panel)
        bpy.types.Scene.da3_input_folder = bpy.props.StringProperty(subtype='DIR_PATH')
    else:
        raise ValueError("installation failed.")

def unregister():
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.unregister_class(operators.DownloadModelOperator)
        bpy.utils.unregister_class(operators.GeneratePointCloudOperator)
        bpy.utils.unregister_class(panels.DA3Panel)
        del bpy.types.Scene.da3_input_folder

if __name__ == "__main__":
    register()