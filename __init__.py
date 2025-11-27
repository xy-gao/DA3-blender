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
        bpy.types.Scene.da3_model_name = bpy.props.EnumProperty(
            items=[
                ('da3-small', 'DA3 Small', 'Small model for faster inference'),
                ('da3-base', 'DA3 Base', 'Base model with balanced performance'),
                ('da3-large', 'DA3 Large', 'Large model for better quality'),
                ('da3-giant', 'DA3 Giant', 'Giant model for highest quality'),
                ('da3mono-large', 'DA3 Mono Large', 'Single image depth estimation'),
                ('da3nested-giant-large', 'DA3 Nested Giant Large', 'Nested depth estimation'),
            ],
            name="Model",
            description="Select DA3 model variant",
            default='da3-large'
        )
        bpy.types.Scene.da3_use_metric = bpy.props.BoolProperty(
            name="Use Metric",
            description="Real-world scale using the metric DA3 model",
            default=False,
        )
        bpy.types.Scene.da3_metric_mode = bpy.props.EnumProperty(
            items=[
                ("scale_base", "Scale Base Depth", "Scale base depth using metric model"),
                ("metric_depth", "Use Metric Depth", "Use metric model depth with base cameras"),
            ],
            name="Metric Mode",
            description="How to combine base and metric model outputs",
            default="scale_base",
        )
    else:
        raise ValueError("installation failed.")

def unregister():
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.unregister_class(operators.DownloadModelOperator)
        bpy.utils.unregister_class(operators.GeneratePointCloudOperator)
        bpy.utils.unregister_class(panels.DA3Panel)
        del bpy.types.Scene.da3_input_folder
        del bpy.types.Scene.da3_model_name
        del bpy.types.Scene.da3_use_metric
        del bpy.types.Scene.da3_metric_mode

if __name__ == "__main__":
    register()