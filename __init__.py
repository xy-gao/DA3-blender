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
import shutil
from pathlib import Path

add_on_path = Path(__file__).parent
DEFAULT_MODELS_DIR = os.path.join(add_on_path, 'models')
_cached_model_folder = None

def get_prefs(context=None):
    if context is None:
        try:
            context = bpy.context
        except AttributeError:
            return None
    package_name = __name__.split('.')[0]
    addon = context.preferences.addons.get(package_name)
    return addon.preferences if addon else None

def get_configured_model_folder(context=None):
    global _cached_model_folder
    prefs = get_prefs(context)
    if prefs:
        _cached_model_folder = getattr(prefs, 'model_folder_path', '')
        return _cached_model_folder
    if _cached_model_folder is not None:
        return _cached_model_folder
    return DEFAULT_MODELS_DIR

class MoveModelsOperator(bpy.types.Operator):
    bl_idname = "da3.move_models"
    bl_label = "Move Models to Custom Folder"
    bl_description = "Move downloaded models from the default folder to the specified custom folder"
    
    def execute(self, context):
        target_dir = get_configured_model_folder(context)
        if target_dir == DEFAULT_MODELS_DIR:
            self.report({'WARNING'}, "Custom model folder is not set or same as default.")
            return {'CANCELLED'}
            
        if not os.path.exists(DEFAULT_MODELS_DIR):
             self.report({'INFO'}, "No default models folder found.")
             return {'FINISHED'}

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        moved_count = 0
        for filename in os.listdir(DEFAULT_MODELS_DIR):
            if filename == ".gitkeep":
                continue
            src_path = os.path.join(DEFAULT_MODELS_DIR, filename)
            if os.path.isfile(src_path):
                dst_path = os.path.join(target_dir, filename)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                else:
                    print(f"Skipping {filename}, already exists in target.")
        
        self.report({'INFO'}, f"Moved {moved_count} files to {target_dir}")
        return {'FINISHED'}

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


class DA3UpdateDA3RepoOperator(bpy.types.Operator):
    bl_idname = "da3.update_da3_repo"
    bl_label = "Update Depth Anything 3"
    bl_description = "Pull latest Depth Anything 3 (recursive) and refresh deps_da3"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        try:
            ok = Dependencies.update_da3_repo()
            if ok:
                self.report({'INFO'}, "Depth Anything 3 updated (git pull --recursive) and deps_da3 refreshed.")
                return {'FINISHED'}
            self.report({'ERROR'}, "Failed to update Depth Anything 3. Check system console.")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to update Depth Anything 3: {e}")
            return {'CANCELLED'}

class DA3AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    model_folder_path: bpy.props.StringProperty(
        name="Model Folder",
        description="Custom folder to store downloaded models. If empty, uses default addon folder.",
        subtype='DIR_PATH',
        default=""
    )

    def draw(self, context):
        layout = self.layout
        
        # Model folder settings
        box = layout.box()
        box.label(text="Model Storage")
        box.prop(self, "model_folder_path")
        
        row = box.row()
        row.operator("da3.move_models", text="Move Existing Models to Custom Folder")

        row = box.row()
        row.operator(DA3UpdateDA3RepoOperator.bl_idname, text="Update Depth Anything 3 (recursive)")
        
        layout.separator()

        if not Dependencies.check():
            layout.label(text="Dependencies are not installed.")
            layout.operator(DA3InstallDepsOperator.bl_idname, text="Install Dependencies")
            layout.label(text="Please check Blender System Console for details.")
        else:
            layout.label(text="Dependencies are installed.")

def register_classes():
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
            ('da3-large-1.1', 'DA3 Large 1.1', 'Large model retrained (preferred)'),
            ('da3-giant-1.1', 'DA3 Giant 1.1', 'Giant model retrained (preferred)'),
            ("da3metric-large", "DA3 Metric Large", "Metric depth model"),
            ('da3mono-large', 'DA3 Mono Large', 'Single image depth estimation'),
            ('da3nested-giant-large-1.1', 'DA3 Nested Giant Large 1.1', 'Nested depth estimation (v1.1 checkpoint)'),
            ('da3-large', 'DA3 Large (obsolete)', 'Large model for better quality'),
            ('da3-giant', 'DA3 Giant (obsolete)', 'Giant model for highest quality'),
            ('da3nested-giant-large', 'DA3 Nested Giant Large (obsolete)', 'Nested depth estimation'),
        ],
        name="Model",
        description="Select DA3 model variant",
        default='da3-large-1.1'
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
    bpy.types.Scene.da3_load_half_precision = bpy.props.BoolProperty(
        name="Use FP16 Weights (experimental)",
        description="Load model weights/buffers in 16-bit to reduce VRAM; may affect stability/accuracy",
        default=False,
    )
    bpy.types.Scene.da3_use_ray_pose = bpy.props.BoolProperty(
        name="Use Ray-based Pose",
        description="Use ray-based camera pose estimation instead of the camera decoder (slower but potentially more accurate)",
        default=False,
    )
    bpy.types.Scene.da3_batch_size = bpy.props.IntProperty(
        name="Batch Size",
        description="Number of images to process in a single batch",
        default=10,
        min=1
    )
    bpy.types.Scene.da3_batch_mode = bpy.props.EnumProperty(
        items=[
            ("ignore_batch_size", "Ignore Batch Size", "Process all images (may use excessive VRAM)"),
            ("skip_frames", "Skip Frames", "Process evenly spaced frames"),
            ("last_frame_overlap", "Last Frame Overlap", "Process overlapping batches for large datasets"),
            ("first_last_overlap", "First+Last Overlap", "Use first and last frame of previous batch plus new frames"),
            ("no_overlap", "No Overlap", "You will have to align batches manually"),
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
    bpy.types.Scene.da3_min_confidence = bpy.props.FloatProperty(
        name="Min Confidence",
        description="Minimum confidence threshold for points (points below this will be removed)",
        default=0.5,
        min=0.0,
        max=100.0,
    )
    bpy.types.Scene.da3_output_debug_images = bpy.props.BoolProperty(
        name="Output Debug Images",
        description="Save debug images (depth, confidence, etc.) to a subfolder",
        default=False,
    )
    bpy.types.Scene.da3_generate_mesh = bpy.props.BoolProperty(
        name="Generate Meshes",
        description="Generate independent textured meshes for each input image instead of a point cloud",
        default=False,
    )
    bpy.types.Scene.da3_detect_motion = bpy.props.BoolProperty(
        name="Detect Motion",
        description="Identify and animate moving objects by checking if they're missing in other frames",
        default=False,
    )
    bpy.types.Scene.da3_motion_threshold = bpy.props.FloatProperty(
        name="Motion Threshold",
        description="Depth difference ratio to consider as empty space (e.g. 0.1 = 10%)",
        default=0.1,
        min=0.01,
        max=1.0,
    )
    bpy.types.Scene.da3_use_segmentation = bpy.props.BoolProperty(
        name="Use Segmentation",
        description="Use YOLO to segment and track objects across frames",
        default=False,
    )
    bpy.types.Scene.da3_segmentation_model = bpy.props.EnumProperty(
        items=[
            ("yolov8n-seg", "YOLOv8 Nano", "Lowest accuracy"),
            ("yolov8l-seg", "YOLOv8 Large", "Balanced speed/accuracy"),
            ("yolov8x-seg", "YOLOv8 X-Large", "Best accuracy for v8"),
            ("yolo11n-seg", "YOLO11 Nano", "Newest tiny fast model"),
            ("yolo11l-seg", "YOLO11 Large", "Newest balanced model"),
            ("yolo11x-seg", "YOLO11 X-Large", "Newest best accuracy"),
            ("yoloe-11s-seg-pf", "YOLOE Small PF", "YOLOE Small prompt-free"),
            ("yoloe-11m-seg-pf", "YOLOE Medium PF", "YOLOE Medium prompt-free"),
            ("yoloe-11l-seg-pf", "YOLOE Large PF", "Recognise the most objects"),
        ],
        name="Seg Model",
        description="Select segmentation model",
        default="yoloe-11l-seg-pf",
    )
    bpy.types.Scene.da3_segmentation_conf = bpy.props.FloatProperty(
        name="Seg Confidence",
        description="Minimum confidence for segmentation",
        default=0.25,
        min=0.0,
        max=1.0,
    )
    bpy.types.Scene.da3_progress = bpy.props.FloatProperty(
        name="Progress",
        subtype='PERCENTAGE',
        default=-1.0,
        min=-1.0,
        max=100.0
    )
    bpy.types.Scene.da3_progress_stage = bpy.props.StringProperty(
        name="Progress Stage",
        default="",
    )

def unregister_classes():
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
    del bpy.types.Scene.da3_use_ray_pose
    del bpy.types.Scene.da3_batch_size
    del bpy.types.Scene.da3_batch_mode
    del bpy.types.Scene.da3_filter_edges
    del bpy.types.Scene.da3_min_confidence
    del bpy.types.Scene.da3_output_debug_images
    del bpy.types.Scene.da3_generate_mesh
    del bpy.types.Scene.da3_detect_motion
    del bpy.types.Scene.da3_motion_threshold
    del bpy.types.Scene.da3_use_segmentation
    del bpy.types.Scene.da3_segmentation_model
    del bpy.types.Scene.da3_segmentation_conf
    del bpy.types.Scene.da3_progress
    del bpy.types.Scene.da3_progress_stage

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

classes = [
    DA3AddonPreferences,
    DA3InstallDepsOperator,
    DA3UpdateDA3RepoOperator,
    MoveModelsOperator,
]

classes_registered = False

def register():
    global classes_registered
    for cls in classes:
        bpy.utils.register_class(cls)

    if Dependencies.check():
        register_classes()
        classes_registered = True
    else:
        bpy.utils.register_class(DA3InstallDepsPanel)

def unregister():
    global classes_registered
    if classes_registered:
        unregister_classes()
    else:
        bpy.utils.unregister_class(DA3InstallDepsPanel)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()