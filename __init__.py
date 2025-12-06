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
import os

def register():
    # Set PyTorch CUDA allocation config to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Set CUDA_LAUNCH_BLOCKING for better error reporting
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    if not Dependencies.check():
        Dependencies.install()
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.register_class(operators.DownloadModelOperator)
        bpy.utils.register_class(operators.UnloadModelOperator)
        bpy.utils.register_class(operators.GeneratePointCloudOperator)
        bpy.utils.register_class(panels.DA3Panel)
        bpy.types.Scene.da3_input_folder = bpy.props.StringProperty(subtype='DIR_PATH')
        bpy.types.Scene.da3_model_name = bpy.props.EnumProperty(
            items=[
                ('da3-small', 'DA3 Small', 'Small model for faster inference'),
                ('da3-base', 'DA3 Base', 'Base model with balanced performance'),
                ('da3-large', 'DA3 Large', 'Large model for better quality'),
                ('da3-giant', 'DA3 Giant', 'Giant model for highest quality'),
                ("da3metric-large", "DA3 Metric Large", "Metric depth model"),
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
        bpy.types.Scene.da3_process_res = bpy.props.IntProperty(
            name="Process Resolution",
            description="Internal resolution for processing (must be multiple of 14)",
            default=504,
            min=14
        )
        bpy.types.Scene.da3_process_res_method = bpy.props.EnumProperty(
            items=[
                ("upper_bound_resize", "Upper Bound Resize", "Resize so that the specified dimension becomes the longer side"),
                ("lower_bound_resize", "Lower Bound Resize", "Resize so that the specified dimension becomes the shorter side"),
            ],
            name="Resize Method",
            description="Method for resizing images to the target resolution",
            default="upper_bound_resize"
        )
        bpy.types.Scene.da3_use_half_precision = bpy.props.BoolProperty(
            name="Use Half Precision",
            description="Use 16-bit floats for reduced VRAM usage",
            default=False,
        )
        bpy.types.Scene.da3_use_ray_pose = bpy.props.BoolProperty(
            name="Use Ray-based Pose",
            description="Use ray-based camera pose estimation instead of the camera decoder (slower but potentially more accurate)",
            default=False,
        )
        bpy.types.Scene.da3_batch_size = bpy.props.IntProperty(
            name="Batch Size",
            description="Number of images to process in batch mode",
            default=10,
            min=1
        )
        bpy.types.Scene.da3_batch_mode = bpy.props.EnumProperty(
            items=[
                ("ignore_batch_size", "Ignore Batch Size", "Process all images (may use excessive VRAM)"),
                ("skip_frames", "Skip Frames", "Process evenly spaced frames"),
                ("last_frame_overlap", "Last Frame Overlap", "Process overlapping batches for large datasets"),
                ("first_last_overlap", "First+Last Overlap", "Use first and last frame of previous batch plus new frames"),
            ],
            name="Batch Mode",
            description="How to select images for processing",
            default="skip_frames"
        )
        bpy.types.Scene.da3_filter_edges = bpy.props.BoolProperty(
            name="Filter Edges",
            description="Set confidence to 0 for pixels with high depth gradient",
            default=True,
        )
        bpy.types.Scene.da3_output_debug_images = bpy.props.BoolProperty(
            name="Output Debug Images",
            description="Save debug images (depth, confidence, etc.) to a subfolder",
            default=False,
        )
    else:
        raise ValueError("installation failed.")

def unregister():
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.unregister_class(operators.DownloadModelOperator)
        bpy.utils.unregister_class(operators.UnloadModelOperator)
        bpy.utils.unregister_class(operators.GeneratePointCloudOperator)
        bpy.utils.unregister_class(panels.DA3Panel)
        del bpy.types.Scene.da3_input_folder
        del bpy.types.Scene.da3_model_name
        del bpy.types.Scene.da3_use_metric
        del bpy.types.Scene.da3_metric_mode
        del bpy.types.Scene.da3_process_res
        del bpy.types.Scene.da3_process_res_method
        del bpy.types.Scene.da3_use_half_precision
        del bpy.types.Scene.da3_use_ray_pose
        del bpy.types.Scene.da3_batch_size
        del bpy.types.Scene.da3_batch_mode
        del bpy.types.Scene.da3_filter_edges
        del bpy.types.Scene.da3_output_debug_images

if __name__ == "__main__":
    register()