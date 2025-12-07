import bpy
from pathlib import Path
import os
import torch
import numpy as np
import time
import datetime
from .utils import (
    run_model,
    convert_prediction_to_dict,
    combine_base_and_metric,
    combine_base_with_metric_depth,
    import_point_cloud,
    create_cameras,
    align_batches,
)

wm = None
total_predicted_time = None
start_time = None
def start_progress_timer(total):
    global wm, total_predicted_time, start_time
    start_time = time.time()
    wm = bpy.context.window_manager
    total_predicted_time = total
    wm.progress_begin(0, 100)
    
    # Calculate estimated duration and finish time
    minutes = int(total // 60)
    seconds = int(total % 60)
    if minutes > 0:
        duration_str = f"{minutes} minutes {seconds} seconds"
    else:
        duration_str = f"{seconds} seconds"
    
    finish_time = datetime.datetime.now() + datetime.timedelta(seconds=total)
    finish_str = finish_time.strftime("%H:%M:%S")
    print(f"Rough estimated duration: {duration_str}, expected finish at {finish_str}")

def update_progress_timer(expected_time, stage=""):
    global wm, total_predicted_time, start_time
    portion = expected_time / total_predicted_time * 100
    wm.progress_update(int(portion))
    print(f"Progress: {stage}, {portion:.2f}%, elapsed: {time.time() - start_time:.2f}s")

def end_progress_timer():
    global wm
    wm.progress_end()
    wm = None

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
        use_ray_pose = getattr(context.scene, "da3_use_ray_pose", False)
        process_res = context.scene.da3_process_res
        process_res_method = context.scene.da3_process_res_method
        use_half_precision = context.scene.da3_use_half_precision
        
        if process_res % 14 != 0:
            self.report({'ERROR'}, "Process resolution must be a multiple of 14.")
            return {'CANCELLED'}
        
        if not input_folder or not os.path.isdir(input_folder):
            self.report({'ERROR'}, "Please select a valid input folder.")
            return {'CANCELLED'}
        
        # Get image paths
        import glob
        image_paths = sorted(glob.glob(os.path.join(input_folder, "*.[jJpP][pPnN][gG]")))
        if not image_paths:
            self.report({'ERROR'}, "No images found in the input folder.")
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

        # Initialize progress bar
        LoadModelTime = 9.2 # seconds
        AlignBatchesTime = 0.29
        AddImagePointsTime = 0.27
        BatchTimePerImage = 4.9 # it's actually quadratic but close enough
        MetricLoadModelTime = 19.25
        MetricBatchTimePerImage = 0.62
        MetricCombineTime = 0.12
        if current_model_name == base_model_name:
            LoadModelTime = 0
        BaseTimeEstimate = LoadModelTime + BatchTimePerImage * len(image_paths) + AlignBatchesTime
        if use_metric:
            MetricTimeEstimate = BaseTimeEstimate + MetricLoadModelTime
            if batch_mode == "scale_base":
                MetricTimeEstimate += MetricBatchTimePerImage * batch_size
            else:
                MetricTimeEstimate += MetricBatchTimePerImage * len(image_paths)
            AfterCombineTimeEstimate = MetricTimeEstimate + AlignBatchesTime + MetricCombineTime
        else:
            MetricTimeEstimate = BaseTimeEstimate
            AfterCombineTimeEstimate = BaseTimeEstimate + AlignBatchesTime
        TotalTimeEstimate = AfterCombineTimeEstimate + AddImagePointsTime*len(image_paths)
        start_progress_timer(TotalTimeEstimate)
        self.report({'INFO'}, "Starting point cloud generation...")

        try:
            # 1) run base model
            self.report({'INFO'}, f"Loading {base_model_name} model...")
            base_model = get_model(base_model_name)
            update_progress_timer(LoadModelTime, "Loaded base model")
            self.report({'INFO'}, "Running base model inference...")
            
            all_base_predictions = []
            
            if batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                # Process in overlapping batches
                if batch_mode == "last_frame_overlap":
                    # Existing scheme: last frame of previous batch overlaps with first of next
                    step = batch_size - 1
                    num_batches = (len(image_paths) + step - 1) // step  # Ceiling division
                    for batch_idx, start_idx in enumerate(range(0, len(image_paths), step)):
                        end_idx = min(start_idx + batch_size, len(image_paths))
                        batch_paths = image_paths[start_idx:end_idx]
                        batch_indices = list(range(start_idx, end_idx))
                        print(f"Batch {batch_idx + 1}/{num_batches}:")
                        prediction = run_model(batch_paths, base_model, process_res, process_res_method, use_half=use_half_precision, use_ray_pose=use_ray_pose)
                        update_progress_timer(LoadModelTime + end_idx * BatchTimePerImage, f"Base batch {batch_idx + 1}")
                        all_base_predictions.append((prediction, batch_indices))
                else:
                    # New scheme: (0..9) (0, 9, 10..17) (10, 17, 18..25)
                    N = len(image_paths)
                    if batch_size < 3:
                        step = 1
                    else:
                        step = batch_size - 2

                    # First batch
                    start = 0
                    end = min(batch_size, N)
                    batch_indices = list(range(start, end))
                    current_new_indices = batch_indices
                    
                    remaining_start = end
                    
                    if step > 0:
                        num_batches = 1 + max(0, (N - end + step - 1) // step)
                    else:
                        num_batches = (N + batch_size - 1) // batch_size

                    batch_idx = 0
                    while True:
                        batch_paths = [image_paths[i] for i in batch_indices]
                        print(f"Batch {batch_idx + 1}/{num_batches}:")
                        prediction = run_model(batch_paths, base_model, process_res, process_res_method, use_half=use_half_precision, use_ray_pose=use_ray_pose)
                        end_idx = batch_indices[-1] + 1
                        update_progress_timer(LoadModelTime + end_idx * BatchTimePerImage, f"Base batch {batch_idx + 1}")
                        all_base_predictions.append((prediction, batch_indices.copy()))

                        if remaining_start >= N:
                            break

                        # Determine overlap frames from the "new" frames of the current batch
                        overlap_indices = [current_new_indices[0], current_new_indices[-1]]
                        # Remove duplicates if any (e.g. if only 1 new frame)
                        if overlap_indices[0] == overlap_indices[1]:
                            overlap_indices = [overlap_indices[0]]

                        next_end = min(remaining_start + step, N)
                        next_new_indices = list(range(remaining_start, next_end))
                        
                        batch_indices = overlap_indices + next_new_indices
                        current_new_indices = next_new_indices
                        
                        remaining_start = next_end
                        batch_idx += 1
            else:
                prediction = run_model(image_paths, base_model, process_res, process_res_method, use_half=use_half_precision, use_ray_pose=use_ray_pose)
                update_progress_timer(LoadModelTime + len(image_paths) * BatchTimePerImage, "Base batch complete")
                all_base_predictions.append((prediction, list(range(len(image_paths)))))
            
            update_progress_timer(BaseTimeEstimate, "Base inference complete")

            # 2) if metric enabled and weights available:
            all_metric_predictions = []
            metric_available = False
            
            if use_metric:
                metric_path = get_model_path("da3metric-large")
                if os.path.exists(metric_path):
                    metric_available = True
                    # free base model from VRAM before loading metric
                    self.report({'INFO'}, "Unloading base model and loading metric model...")
                    base_model = None
                    unload_current_model()

                    metric_model = get_model("da3metric-large")
                    update_progress_timer(BaseTimeEstimate + MetricLoadModelTime, "Loaded metric model")
                    self.report({'INFO'}, "Running metric model inference...")
                    
                    if metric_mode == "scale_base":
                        # In scale_base mode, run **one** metric batch over all images.
                        N = len(image_paths)
                        start = 0
                        end = min(batch_size, N)
                        batch_indices = list(range(start, end))
                        batch_paths = [image_paths[i] for i in batch_indices]
                        prediction = run_model(
                            batch_paths,
                            metric_model,
                            process_res,
                            process_res_method,
                            use_half=use_half_precision,
                            use_ray_pose=use_ray_pose,
                        )
                        update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + end * MetricBatchTimePerImage, "Metric batch complete")
                        all_metric_predictions.append((prediction, batch_indices.copy()))
                    else:
                        # For other metric modes, keep previous batching behaviour
                        if batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                            # Process in overlapping batches for metric too (mirror base logic)
                            if batch_mode == "last_frame_overlap":
                                step = batch_size - 1
                                num_batches = (len(image_paths) + step - 1) // step
                                for batch_idx, start_idx in enumerate(range(0, len(image_paths), step)):
                                    end_idx = min(start_idx + batch_size, len(image_paths))
                                    batch_paths = image_paths[start_idx:end_idx]
                                    batch_indices = list(range(start_idx, end_idx))
                                    print(f"Batch {batch_idx + 1}/{num_batches}:")
                                    prediction = run_model(batch_paths, metric_model, process_res, process_res_method, use_half=use_half_precision, use_ray_pose=use_ray_pose)
                                    update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + end_idx * MetricBatchTimePerImage, f"Metric batch {batch_idx + 1}")
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
                                current_new_indices = batch_indices
                                
                                remaining_start = end
                                
                                if step > 0:
                                    num_batches = 1 + max(0, (N - end + step - 1) // step)
                                else:
                                    num_batches = (N + batch_size - 1) // batch_size

                                batch_idx = 0
                                while True:
                                    batch_paths = [image_paths[i] for i in batch_indices]
                                    print(f"Batch {batch_idx + 1}/{num_batches}:")
                                    prediction = run_model(batch_paths, metric_model, process_res, process_res_method, use_half=use_half_precision, use_ray_pose=use_ray_pose)
                                    end_idx = batch_indices[-1] + 1
                                    update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + end_idx * MetricBatchTimePerImage, f"Metric batch {batch_idx + 1}")
                                    all_metric_predictions.append((prediction, batch_indices.copy()))

                                    if remaining_start >= N:
                                        break

                                    overlap_indices = [current_new_indices[0], current_new_indices[-1]]
                                    if overlap_indices[0] == overlap_indices[1]:
                                        overlap_indices = [overlap_indices[0]]

                                    next_end = min(remaining_start + step, N)
                                    next_new_indices = list(range(remaining_start, next_end))
                                    
                                    batch_indices = overlap_indices + next_new_indices
                                    current_new_indices = next_new_indices
                                    
                                    remaining_start = next_end
                                    batch_idx += 1
                        else:
                            # Non-overlapping full batch
                            prediction = run_model(image_paths, metric_model, process_res, process_res_method, use_half=use_half_precision, use_ray_pose=use_ray_pose)
                            all_metric_predictions.append((prediction, list(range(len(image_paths)))))
                            update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + len(image_paths) * MetricBatchTimePerImage, "Metric batch complete")
                    metric_model = None
                    unload_current_model()
                else:
                    self.report({'WARNING'}, "Metric model not downloaded; using non-metric depth only.")
            
            
            update_progress_timer(MetricTimeEstimate, "Metric inference complete")
            # Align base batches. Metric is **not** aligned in scale_base mode.
            if batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                aligned_base_predictions = align_batches(all_base_predictions)
                # Metric depth is absolute, and has no camera poses, so alignment between batches is less important (and not implemented yet).
                if metric_available:
                    aligned_metric_predictions = [p[0] for p in all_metric_predictions]
            else:
                aligned_base_predictions = [p[0] for p in all_base_predictions]
                if metric_available:
                    aligned_metric_predictions = [p[0] for p in all_metric_predictions]
            update_progress_timer(MetricTimeEstimate + AlignBatchesTime, "Align batches complete")

            # Create or get a collection named after the folder
            folder_name = os.path.basename(os.path.normpath(input_folder))
            scene = context.scene
            collections = bpy.data.collections
            
            # Create parent collection
            parent_col = collections.new(folder_name)
            scene.collection.children.link(parent_col)

            # Combine the base and metric predictions
            if metric_available:
                all_combined_predictions = combine_base_and_metric(aligned_base_predictions, aligned_metric_predictions)
            else:
                all_combined_predictions = aligned_base_predictions
            update_progress_timer(AfterCombineTimeEstimate, "Combined predictions complete")

            # Add a point cloud for each batch
            for batch_number, batch_prediction in enumerate(all_combined_predictions):
                batch_indices = all_base_predictions[batch_number][1]
                batch_paths = [image_paths[j] for j in batch_indices]
                
                combined_predictions = convert_prediction_to_dict(batch_prediction, batch_paths)
                
                # Create batch collection
                batch_col_name = f"{folder_name}_Batch_{batch_number+1}"
                batch_col = collections.new(batch_col_name)
                parent_col.children.link(batch_col)
                
                import_point_cloud(combined_predictions, collection=batch_col)
                create_cameras(combined_predictions, collection=batch_col)
                end_idx = batch_indices[-1] + 1
                update_progress_timer(AfterCombineTimeEstimate + AddImagePointsTime * end_idx, f"Added batch {batch_number + 1} to Blender")
            
            update_progress_timer(TotalTimeEstimate, "Point cloud generation complete")
            end_progress_timer()
            self.report({'INFO'}, "Point cloud generation complete.")
        except Exception as e:
            end_progress_timer()
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