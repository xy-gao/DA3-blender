import bpy
from pathlib import Path
import os
import torch
import numpy as np
from .utils import run_model, import_point_cloud, create_cameras
import urllib.request
import threading

add_on_path = Path(__file__).parent
MODELS_DIR = os.path.join(add_on_path, 'models')
_URLS = {
    'da3-small': "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors",
    'da3-base': "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors",
    'da3-large': "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors",
    'da3-giant': "https://huggingface.co/depth-anything/DA3-GIANT/resolve/main/model.safetensors",
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

class DownloadModelOperator(bpy.types.Operator):
    bl_idname = "da3.download_model"
    bl_label = "Download DA3 Model"

    thread = None
    progress = 0
    error_message = ""
    stop_event = None

    def invoke(self, context, event):
        model_name = context.scene.da3_model_name
        self.model_path = get_model_path(model_name)
        
        if os.path.exists(self.model_path):
            self.report({'INFO'}, f"Model {model_name} already downloaded.")
            return {'FINISHED'}
        
        if model_name not in _URLS:
            self.report({'ERROR'}, f"Unknown model: {model_name}")
            return {'CANCELLED'}

        context.scene.da3_progress = 0
        self.progress = 0
        self.error_message = ""
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self.download_file, args=(_URLS[model_name], self.model_path, context))
        self.thread.start()

        self.timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def download_file(self, url, path, context):
        try:
            print(f"Downloading model {url.split('/')[-2]}...")
            os.makedirs(MODELS_DIR, exist_ok=True)

            with urllib.request.urlopen(url) as response:
                total_size = int(response.info().get('Content-Length', -1))
                chunk_size = 1024 * 4
                bytes_so_far = 0

                with open(path, 'wb') as f:
                    while not self.stop_event.is_set():
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_so_far += len(chunk)

                        if total_size > 0:
                            self.progress = (bytes_so_far / total_size) * 100
                            context.scene.da3_progress = self.progress

            if not self.stop_event.is_set():
                 self.progress = 100
                 context.scene.da3_progress = 100

        except Exception as e:
            self.error_message = f"Failed to download model: {e}"
        finally:
            self.stop_event.set()

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self.stop_event.is_set() and not self.thread.is_alive():
                context.window_manager.event_timer_remove(self.timer)
                context.scene.da3_progress = 0

                if self.error_message:
                    self.report({'ERROR'}, self.error_message)
                    if os.path.exists(self.model_path):
                        os.remove(self.model_path)
                    return {'CANCELLED'}

                model_name = context.scene.da3_model_name
                self.report({'INFO'}, f"Model {model_name} downloaded successfully.")

                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

                return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.stop_event.set()
            context.scene.da3_progress = 0

            wm = context.window_manager
            wm.event_timer_remove(self.timer)

            self.report({'WARNING'}, "Download cancelled.")

            if self.thread.is_alive():
                self.thread.join()

            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    @classmethod
    def poll(cls, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name)
        return not os.path.exists(model_path)


class GeneratePointCloudOperator(bpy.types.Operator):
    bl_idname = "da3.generate_point_cloud"
    bl_label = "Generate Point Cloud"

    thread = None
    progress = 0
    error_message = ""
    stop_event = None

    def invoke(self, context, event):
        self.input_folder = context.scene.da3_input_folder
        self.model_name = context.scene.da3_model_name
        self.batch_size = context.scene.da3_batch_size
        
        if not self.input_folder or not os.path.isdir(self.input_folder):
            self.report({'ERROR'}, "Please select a valid input folder.")
            return {'CANCELLED'}

        context.scene.da3_progress = 0
        self.progress = 0
        self.error_message = ""
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self.generate_worker, args=(context,))
        self.thread.start()

        self.timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def generate_worker(self, context):
        try:
            model = get_model(self.model_name)

            def progress_callback(progress_value):
                if self.stop_event.is_set():
                    return True
                self.progress = progress_value
                context.scene.da3_progress = progress_value
                return False

            self.predictions = run_model(self.input_folder, model, self.batch_size, progress_callback)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_message = f"Failed to generate point cloud: {e}"
        finally:
            self.stop_event.set()

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self.stop_event.is_set() and not self.thread.is_alive():
                context.window_manager.event_timer_remove(self.timer)
                context.scene.da3_progress = 0

                if self.error_message:
                    self.report({'ERROR'}, self.error_message)
                    return {'CANCELLED'}

                if self.predictions:
                    import_point_cloud(self.predictions)
                    create_cameras(self.predictions)
                    self.report({'INFO'}, "Point cloud generated and imported successfully.")
                    self.report({'INFO'}, "Cameras generated successfully.")
                return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.stop_event.set()
            context.scene.da3_progress = 0

            context.window_manager.event_timer_remove(self.timer)

            self.report({'WARNING'}, "Generation cancelled.")

            if self.thread.is_alive():
                self.thread.join()

            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    @classmethod
    def poll(cls, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name)
        return os.path.exists(model_path) and context.scene.da3_input_folder != ""