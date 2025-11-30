import bpy
from pathlib import Path
import os
import torch
import numpy as np
from .utils import (
    run_model,
    convert_prediction_to_dict,
    combine_base_and_metric,
    combine_base_with_metric_depth,
    import_point_cloud,
    create_cameras,
    combine_overlapping_predictions,
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

def display_VRAM_usage(stage: str, include_peak=False):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        free, total = torch.cuda.mem_get_info()
        free_mb = free / 1024**2
        total_mb = total / 1024**2
        msg = f"VRAM {stage}: {allocated:.1f} MB (free: {free_mb:.1f} MB / {total_mb:.1f} MB)"
        if include_peak:
            peak = torch.cuda.max_memory_allocated() / 1024**2
            msg += f" (peak: {peak:.1f} MB)"
        print(msg)


def get_model(model_name):
    global model, current_model_name
    if model is None or current_model_name != model_name:
        from depth_anything_3.api import DepthAnything3
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        display_VRAM_usage(f"before loading {model_name}")
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
        display_VRAM_usage(f"after loading {model_name}", include_peak=True)
    return model

def unload_current_model():
    global model, current_model_name
    if model is not None:
        display_VRAM_usage("before unload")
        # Drop references so PyTorch can free memory
        del model
        model = None
        current_model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            display_VRAM_usage("after unload")

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


class UnloadModelOperator(bpy.types.Operator):
    bl_idname = "da3.unload_model"
    bl_label = "Unload Model"

    def execute(self, context):
        unload_current_model()
        self.report({'INFO'}, "Model unloaded and VRAM freed.")
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        # Enable if a model is loaded
        return model is not None


class GeneratePointCloudOperator(bpy.types.Operator):
    bl_idname = "da3.generate_point_cloud"
    bl_label = "Generate Point Cloud"

    def execute(self, context):
        input_folder = context.scene.da3_input_folder
        base_model_name = context.scene.da3_model_name
        use_metric = context.scene.da3_use_metric
        metric_mode = getattr(context.scene, "da3_metric_mode", "scale_base")
        process_res = context.scene.da3_process_res
        process_res_method = context.scene.da3_process_res_method
        use_half_precision = context.scene.da3_use_half_precision
        
        if process_res % 14 != 0:
            self.report({'ERROR'}, "Process resolution must be a multiple of 14.")
            return {'CANCELLED'}
        
        # Warn about potential timeout with high resolution
        if process_res > 400:
            self.report({'WARNING'}, "High process resolution may cause GPU timeout. Consider reducing to 350-392 for stability.")
        
        if not input_folder or not os.path.isdir(input_folder):
            self.report({'ERROR'}, "Please select a valid input folder.")
            return {'CANCELLED'}
        
        # Initialize progress bar
        wm = bpy.context.window_manager
        wm.progress_begin(0, 100)
        self.report({'INFO'}, "Starting point cloud generation...")
        
        # Get image paths
        import glob
        image_paths = sorted(glob.glob(os.path.join(input_folder, "*.[jJpP][pPnN][gG]")))
        if not image_paths:
            self.report({'ERROR'}, "No images found in the input folder.")
            wm.progress_end()
            return {'CANCELLED'}
        
        print(f"Total images: {len(image_paths)}")
        
        batch_mode = context.scene.da3_batch_mode
        batch_size = context.scene.da3_batch_size
        if batch_mode == "skip_frames" and len(image_paths) > batch_size:
            import numpy as np
            indices = np.linspace(0, len(image_paths) - 1, batch_size, dtype=int)
            image_paths = [image_paths[i] for i in indices]
        # For overlap modes and ignore_batch_size, use all images
        
        self.report({'INFO'}, f"Processing {len(image_paths)} images...")
        
        try:
            # 1) run base model
            wm.progress_update(5)
            self.report({'INFO'}, f"Loading {base_model_name} model...")
            base_model = get_model(base_model_name)
            wm.progress_update(15)
            self.report({'INFO'}, "Running base model inference...")
            
            if batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                # Process in overlapping batches
                all_base_predictions = []

                if batch_mode == "last_frame_overlap":
                    # Existing scheme: last frame of previous batch overlaps with first of next
                    step = batch_size - 1
                    num_batches = (len(image_paths) + step - 1) // step  # Ceiling division
                    for batch_idx, start_idx in enumerate(range(0, len(image_paths), step)):
                        end_idx = min(start_idx + batch_size, len(image_paths))
                        batch_paths = image_paths[start_idx:end_idx]
                        batch_indices = list(range(start_idx, end_idx))
                        print(f"Batch {batch_idx + 1}/{num_batches}:")
                        prediction = run_model(batch_paths, base_model, process_res, process_res_method, use_half=use_half_precision)
                        all_base_predictions.append((prediction, batch_indices))
                else:
                    # New scheme: [prev_first_new, prev_last_new] + N new frames
                    N = len(image_paths)
                    if batch_size < 3:
                        step = 1
                    else:
                        step = batch_size - 2

                    # First batch: just the first batch_size frames
                    start = 0
                    end = min(batch_size, N)
                    batch_indices = list(range(start, end))
                    num_batches = 1
                    remaining_start = end
                    # Rough upper bound for logging
                    if step > 0:
                        num_batches = 1 + max(0, (N - end + step - 1) // step)

                    batch_idx = 0
                    while True:
                        # Build batch paths from indices
                        batch_paths = [image_paths[i] for i in batch_indices]
                        print(f"Batch {batch_idx + 1}/{num_batches}:")
                        prediction = run_model(batch_paths, base_model, process_res, process_res_method, use_half=use_half_precision)
                        all_base_predictions.append((prediction, batch_indices.copy()))

                        if remaining_start >= N:
                            break

                        # Define new first/last from current batch (its first and last new frames)
                        first_new = batch_indices[0]
                        last_new = batch_indices[-1]

                        # Next batch indices: [first_new, last_new] + next unseen indices
                        next_indices = list(range(remaining_start, min(remaining_start + step, N)))
                        batch_indices = [first_new, last_new] + next_indices

                        remaining_start += len(next_indices)
                        batch_idx += 1

                base_prediction = combine_overlapping_predictions(all_base_predictions, image_paths)
            else:
                base_prediction = run_model(image_paths, base_model, process_res, process_res_method, use_half=use_half_precision)
            
            wm.progress_update(60)

            # 2) if metric enabled and weights available:
            if use_metric:
                metric_path = get_model_path("da3metric-large")
                if os.path.exists(metric_path):
                    # free base model from VRAM before loading metric
                    wm.progress_update(65)
                    self.report({'INFO'}, "Unloading base model and loading metric model...")
                    base_model = None
                    unload_current_model()

                    metric_model = get_model("da3metric-large")
                    wm.progress_update(75)
                    self.report({'INFO'}, "Running metric model inference...")
                    
                    if batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                        # Process in overlapping batches for metric too (mirror base logic)
                        all_metric_predictions = []

                        if batch_mode == "last_frame_overlap":
                            step = batch_size - 1
                            num_batches = (len(image_paths) + step - 1) // step
                            for batch_idx, start_idx in enumerate(range(0, len(image_paths), step)):
                                end_idx = min(start_idx + batch_size, len(image_paths))
                                batch_paths = image_paths[start_idx:end_idx]
                                batch_indices = list(range(start_idx, end_idx))
                                print(f"Batch {batch_idx + 1}/{num_batches}:")
                                prediction = run_model(batch_paths, metric_model, process_res, process_res_method, use_half=use_half_precision)
                                all_metric_predictions.append((prediction, batch_indices))
                        else:
                            N = len(image_paths)
                            if batch_size < 3:
                                step = 1
                            else:
                                step = batch_size - 2

                            start = 0
                            end = min(batch_size, N)
                            batch_indices = list(range(start, end))
                            num_batches = 1
                            remaining_start = end
                            if step > 0:
                                num_batches = 1 + max(0, (N - end + step - 1) // step)

                            batch_idx = 0
                            while True:
                                batch_paths = [image_paths[i] for i in batch_indices]
                                print(f"Batch {batch_idx + 1}/{num_batches}:")
                                prediction = run_model(batch_paths, metric_model, process_res, process_res_method, use_half=use_half_precision)
                                all_metric_predictions.append((prediction, batch_indices.copy()))

                                if remaining_start >= N:
                                    break

                                first_new = batch_indices[0]
                                last_new = batch_indices[-1]
                                next_indices = list(range(remaining_start, min(remaining_start + step, N)))
                                batch_indices = [first_new, last_new] + next_indices

                                remaining_start += len(next_indices)
                                batch_idx += 1

                        metric_prediction = combine_overlapping_predictions(all_metric_predictions, image_paths)
                    else:
                        metric_prediction = run_model(image_paths, metric_model, process_res, process_res_method, use_half=use_half_precision)
                    
                    wm.progress_update(90)
                    metric_model = None
                    unload_current_model()

                    if metric_mode == "metric_depth":
                        combined_prediction = combine_base_with_metric_depth(
                            base_prediction, metric_prediction
                        )
                    else:
                        combined_prediction = combine_base_and_metric(
                            base_prediction, metric_prediction
                        )
                    combined_predictions = convert_prediction_to_dict(combined_prediction, image_paths)
                else:
                    self.report({'WARNING'}, "Metric model not downloaded; using non-metric depth only.")
                    combined_predictions = convert_prediction_to_dict(base_prediction, image_paths)
            else:
                combined_predictions = convert_prediction_to_dict(base_prediction, image_paths)

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
            
            wm.progress_update(100)
            wm.progress_end()
            self.report({'INFO'}, "Point cloud generation complete.")
        except Exception as e:
            wm.progress_end()
            import traceback
            print("DA3 ERROR while generating point cloud:")
            traceback.print_exc()
            base_model = None
            metric_model = None
            base_prediction = None
            metric_prediction = None
            combined_prediction = None
            combined_predictions = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()  # Force free any pending allocations
                except Exception as e:
                    print(f"Warning: Failed to empty CUDA cache: {e}")
            import gc
            gc.collect()  # Force garbage collection
            unload_current_model()  # Free VRAM on error
            self.report({'ERROR'}, f"Failed to generate point cloud: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name)
        return os.path.exists(model_path) and context.scene.da3_input_folder != ""