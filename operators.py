import bpy
from pathlib import Path
import os
import torch
import numpy as np
from .utils import (
    run_single_model,
    convert_prediction_to_dict,
    combine_base_and_metric,
    combine_base_with_metric_depth,
    import_point_cloud,
    create_cameras,
)

add_on_path = Path(__file__).parent
MODELS_DIR = os.path.join(add_on_path, 'models')
_URLS = {
    'da3-small': "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors",
    'da3-base': "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors",
    'da3-large': "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors",
    'da3-giant': "https://huggingface.co/depth-anything/DA3-GIANT/resolve/main/model.safetensors",
    "da3metric-large": "https://huggingface.co/depth-anything/DA3METRIC-LARGE/resolve/main/model.safetensors",
    "da3mono-large": "https://huggingface.co/depth-anything/DA3MONO-LARGE/resolve/main/model.safetensors",
    "da3nested-giant-large": "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE/resolve/main/model.safetensors",
}
model = None
current_model_name = None

def get_model_path(model_name):
    return os.path.join(MODELS_DIR, f'{model_name}.safetensors')

def get_model(model_name):
    global model, current_model_name
    if model is None or current_model_name != model_name:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3(model_name=model_name)
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            from safetensors.torch import load_file
            weight = load_file(model_path)
            model.load_state_dict(weight, strict=False)
        else:
            raise FileNotFoundError(f"Model file {model_name} not found. Please download it first.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        current_model_name = model_name
    return model

def unload_current_model():
    global model, current_model_name
    if model is not None:
        # Drop references so PyTorch can free memory
        del model
        model = None
        current_model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class DownloadModelOperator(bpy.types.Operator):
    bl_idname = "da3.download_model"
    bl_label = "Download DA3 Model"

    # NEW: optional override for which model to download
    da3_override_model_name: bpy.props.StringProperty(
        name="Override Model Name",
        description="If set, download this model instead of the one selected in the scene",
        default="",
    )

    def execute(self, context):
        model_name = self.da3_override_model_name or context.scene.da3_model_name
        model_path = get_model_path(model_name)
        
        if os.path.exists(model_path):
            self.report({'INFO'}, f"Model {model_name} already downloaded.")
            return {'FINISHED'}
        
        if model_name not in _URLS:
            self.report({'ERROR'}, f"Unknown model: {model_name}")
            return {'CANCELLED'}
            
        try:
            print(f"Downloading model {model_name}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.hub.download_url_to_file(_URLS[model_name], model_path)
            self.report({'INFO'}, f"Model {model_name} downloaded successfully.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to download model {model_name}: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        # Allow the button to be clicked; existence is checked in execute()
        return True
        # model_name = context.scene.da3_model_name
        # model_path = get_model_path(model_name)
        # return not os.path.exists(model_path)


class GeneratePointCloudOperator(bpy.types.Operator):
    bl_idname = "da3.generate_point_cloud"
    bl_label = "Generate Point Cloud"

    def execute(self, context):
        input_folder = context.scene.da3_input_folder
        base_model_name = context.scene.da3_model_name
        use_metric = context.scene.da3_use_metric
        metric_mode = getattr(context.scene, "da3_metric_mode", "scale_base")
        
        if not input_folder or not os.path.isdir(input_folder):
            self.report({'ERROR'}, "Please select a valid input folder.")
            return {'CANCELLED'}
        try:
            # 1) run base model
            base_model = get_model(base_model_name)
            base_prediction, base_image_paths = run_single_model(input_folder, base_model)

            # 2) if metric enabled and weights available:
            if use_metric:
                metric_path = get_model_path("da3metric-large")
                if os.path.exists(metric_path):
                    # free base model from VRAM before loading metric
                    unload_current_model()

                    metric_model = get_model("da3metric-large")
                    metric_prediction, metric_image_paths = run_single_model(input_folder, metric_model)

                    if metric_mode == "metric_depth":
                        combined_prediction = combine_base_with_metric_depth(
                            base_prediction, metric_prediction
                        )
                    else:
                        combined_prediction = combine_base_and_metric(
                            base_prediction, metric_prediction
                        )
                    combined_predictions = convert_prediction_to_dict(combined_prediction, base_image_paths)
                else:
                    self.report({'WARNING'}, "Metric model not downloaded; using non-metric depth only.")
                    combined_predictions = convert_prediction_to_dict(base_prediction, base_image_paths)
            else:
                combined_predictions = convert_prediction_to_dict(base_prediction, base_image_paths)

            # Create or get a collection named after the folder
            folder_name = os.path.basename(os.path.normpath(input_folder))
            scene = context.scene
            collections = bpy.data.collections
            target_col = collections.new(folder_name)
            scene.collection.children.link(target_col)

            # 3) import point cloud and create cameras
            import_point_cloud(combined_predictions, collection=target_col)
            self.report({'INFO'}, "Point cloud generated and imported successfully.")
            create_cameras(combined_predictions, collection=target_col)
            self.report({'INFO'}, "Cameras generated successfully.")
        except Exception as e:
            import traceback
            print("DA3 ERROR while generating point cloud:")
            traceback.print_exc()
            self.report({'ERROR'}, f"Failed to generate point cloud: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name)
        return os.path.exists(model_path) and context.scene.da3_input_folder != ""