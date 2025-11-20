import bpy
from pathlib import Path
import os
import torch
import numpy as np
from .utils import run_model, import_point_cloud, create_cameras

add_on_path = Path(__file__).parent
MODELS_DIR = os.path.join(add_on_path, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.safetensors')
_URL = "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors"
model = None

def get_model():
    global model
    if model is None:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3()
        if os.path.exists(MODEL_PATH):
            from safetensors.torch import load_file
            weight = load_file(MODEL_PATH)
            model.load_state_dict(weight, strict=False)
        else:
            raise FileNotFoundError("Model file not found. Please download it first.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
    return model

class DownloadModelOperator(bpy.types.Operator):
    bl_idname = "da3.download_model"
    bl_label = "Download DA3 Model"

    def execute(self, context):
        if os.path.exists(MODEL_PATH):
            self.report({'INFO'}, "Model already downloaded.")
            return {'FINISHED'}
        try:
            print("downloading model...")
            torch.hub.download_url_to_file(_URL, MODEL_PATH)
            self.report({'INFO'}, "Model downloaded successfully.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to download model: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        return not os.path.exists(MODEL_PATH)


class GeneratePointCloudOperator(bpy.types.Operator):
    bl_idname = "da3.generate_point_cloud"
    bl_label = "Generate Point Cloud"

    def execute(self, context):
        input_folder = context.scene.da3_input_folder
        if not input_folder or not os.path.isdir(input_folder):
            self.report({'ERROR'}, "Please select a valid input folder.")
            return {'CANCELLED'}
        try:
            model = get_model()
            predictions = run_model(input_folder, model)
            import_point_cloud(predictions)
            self.report({'INFO'}, "Point cloud generated and imported successfully.")
            create_cameras(predictions)
            self.report({'INFO'}, "Cameras generated successfully.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate point cloud: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        return os.path.exists(MODEL_PATH) and context.scene.da3_input_folder != ""