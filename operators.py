import bpy
from pathlib import Path
import os
import shutil
import torch
from collections import Counter
import numpy as np
import time
import datetime
import subprocess
import sys
import tempfile

from . import streaming as streaming_runner
from .utils import (
    run_model,
    convert_prediction_to_dict,
    combine_base_and_metric,
    combine_base_with_metric_depth,
    import_point_cloud,
    import_mesh_from_depth,
    create_point_cloud_object,
    create_cameras,
    align_batches,
    align_single_batch,
    compute_motion_scores,
)
import urllib.request
import threading
import queue
from . import DEFAULT_MODELS_DIR, get_configured_model_folder, get_prefs
import types


def get_any_model_path(model_filename, context=None):
    # Check configured folder first
    configured_folder = get_configured_model_folder(context)
        
    configured_path = os.path.join(configured_folder, model_filename)
    if os.path.exists(configured_path):
        return configured_path
        
    # Check default folder
    default_path = os.path.join(DEFAULT_MODELS_DIR, model_filename)
    if os.path.exists(default_path):
        return default_path
        
    # If not found, return configured path for download
    return configured_path

_URLS = {
    'da3-small': "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors",
    'da3-base': "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors",
    'da3-large': "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors",
    'da3-large-1.1': "https://huggingface.co/depth-anything/DA3-LARGE-1.1/resolve/main/model.safetensors",
    'da3-giant': "https://huggingface.co/depth-anything/DA3-GIANT/resolve/main/model.safetensors",
    'da3-giant-1.1': "https://huggingface.co/depth-anything/DA3-GIANT-1.1/resolve/main/model.safetensors",
    "da3metric-large": "https://huggingface.co/depth-anything/DA3METRIC-LARGE/resolve/main/model.safetensors",
    "da3mono-large": "https://huggingface.co/depth-anything/DA3MONO-LARGE/resolve/main/model.safetensors",
    "da3nested-giant-large": "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE/resolve/main/model.safetensors",
    "da3nested-giant-large-1.1": "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/model.safetensors",

    "dino_salad": "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",

    "yolov8n-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
    "yolov8s-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt",
    "yolov8m-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt",
    "yolov8l-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-seg.pt",
    "yolov8x-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-seg.pt",
    "yolo11n-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
    "yolo11s-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
    "yolo11m-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt",
    "yolo11l-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
    "yolo11x-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",
    "yoloe-11s-seg-pf": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt",
    "yoloe-11m-seg-pf": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg-pf.pt",
    "yoloe-11l-seg-pf": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg-pf.pt",
}
NESTED_MODEL_NAMES = ("da3nested-giant-large", "da3nested-giant-large-1.1")
CONFIG_NAME_MAP = {
    # Map alt checkpoint labels to the existing config stem
    "da3-large-1.1": "da3-large",
    "da3-giant-1.1": "da3-giant",
    "da3nested-giant-large-1.1": "da3nested-giant-large",
}

ADDON_PATH = Path(__file__).parent
STREAMING_DIR = ADDON_PATH / "da3_repo" / "da3_streaming"
STREAMING_SCRIPT = STREAMING_DIR / "da3_streaming.py"
STREAMING_CONFIGS = {
    "base_config_low_vram.yaml": STREAMING_DIR / "configs" / "base_config_low_vram.yaml",
    "base_config.yaml": STREAMING_DIR / "configs" / "base_config.yaml",
    "kitti.yaml": STREAMING_DIR / "configs" / "kitti.yaml",
    "tum.yaml": STREAMING_DIR / "configs" / "tum.yaml",
}
DEPS_PUBLIC_PATH = ADDON_PATH / "deps_public"
DEPS_DA3_PATH = ADDON_PATH / "deps_da3"
model = None
current_model_name = None
current_model_load_half = None

def get_model_path(model_name, context=None):
    return get_any_model_path(f'{model_name}.safetensors', context)

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


def unload_if_overcommitted(reason: str = ""):
    """Unload the cached model if current VRAM usage exceeds the device's total VRAM."""
    if not torch.cuda.is_available():
        return False
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    if allocated > total:
        total_mb = total / 1024**2
        alloc_mb = allocated / 1024**2
        reserv_mb = reserved / 1024**2
        print(
            f"VRAM overcommitted (allocated: {alloc_mb:.1f} MB, reserved: {reserv_mb:.1f} MB, total: {total_mb:.1f} MB). "
            f"Unloading model. Reason: {reason}"
        )
        unload_current_model()
        torch.cuda.empty_cache()
        return True
    return False


def _summarize_model_dtypes(mod):
    # Lightweight dtype/device counter for debugging footprint
    counts = Counter()
    for p in mod.parameters():
        counts[(p.dtype, p.device)] += p.numel()
    for b in mod.buffers():
        counts[(b.dtype, b.device)] += b.numel()
    return counts


def _convert_norm_layers(model, dtype):
    # Ensure normalization weights/bias match the requested dtype to avoid mixed-dtype layer_norm errors
    norm_types = (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    )
    for m in model.modules():
        if isinstance(m, norm_types):
            m.to(dtype)


def _register_norm_input_cast_hooks(model):
    # Cast norm inputs to the module's weight dtype to avoid runtime dtype mismatches
    norm_types = (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    )

    def make_hook(mod):
        def hook(module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if isinstance(x, torch.Tensor) and module.weight is not None:
                target_dtype = module.weight.dtype
                if x.dtype != target_dtype:
                    return (x.to(target_dtype),) + inputs[1:]
            return inputs
        return hook

    for m in model.modules():
        if isinstance(m, norm_types):
            m.register_forward_pre_hook(make_hook(m))


def _register_conv_input_cast_hooks(model):
    # Cast Conv/ConvTranspose inputs to the module's weight dtype to avoid float/half mismatches
    conv_types = (
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
    )

    def make_hook(mod):
        def hook(module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if isinstance(x, torch.Tensor) and module.weight is not None:
                target_dtype = module.weight.dtype
                if x.dtype != target_dtype:
                    return (x.to(target_dtype),) + inputs[1:]
            return inputs
        return hook

    for m in model.modules():
        if isinstance(m, conv_types):
            m.register_forward_pre_hook(make_hook(m))


def _register_linear_input_cast_hooks(model):
    # Cast Linear inputs to the module's weight dtype to avoid matmul dtype mismatches
    def make_hook(mod):
        def hook(module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if isinstance(x, torch.Tensor) and module.weight is not None:
                target_dtype = module.weight.dtype
                if x.dtype != target_dtype:
                    return (x.to(target_dtype),) + inputs[1:]
            return inputs
        return hook

    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_pre_hook(make_hook(m))


def _patch_nested_alignment_fp16(model):
    """
    Monkeypatch nested giant depth alignment to avoid fp16 quantile crash without modifying deps on disk.
    Converts sampled confidence to float before torch.quantile.
    """
    try:
        from depth_anything_3.utils.alignment import (
            compute_sky_mask,
            compute_alignment_mask,
            least_squares_scale_scalar,
            sample_tensor_for_quantile,
        )
    except Exception as e:
        print(f"Warning: could not import alignment utils for nested patch: {e}")
        return

    da3_core = getattr(model, "model", None)
    if da3_core is None:
        return

    # Save original for fallback
    orig_fn = getattr(da3_core, "_apply_depth_alignment", None)
    if orig_fn is None:
        return

    def _patched(self, output, metric_output):
        # Run alignment in fp32 without autocast to avoid scalar type issues
        with torch.cuda.amp.autocast(enabled=False):
            non_sky_mask = compute_sky_mask(metric_output.sky.float(), threshold=0.3)
            assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"
            depth_conf_ns = output.depth_conf.float()[non_sky_mask]
            depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
            median_conf = torch.quantile(depth_conf_sampled, 0.5)
            align_mask = compute_alignment_mask(
                output.depth_conf.float(), non_sky_mask, output.depth.float(), metric_output.depth.float(), median_conf
            )
            valid_depth = output.depth.float()[align_mask]
            valid_metric_depth = metric_output.depth.float()[align_mask]
            scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)
        output.depth *= scale_factor
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()
        return output

    da3_core._apply_depth_alignment = types.MethodType(_patched, da3_core)


def _patch_nested_sky_fp16(model):
    """
    Monkeypatch nested giant sky handling to run quantile in fp32 without modifying deps on disk.
    """
    try:
        from depth_anything_3.utils.alignment import set_sky_regions_to_max_depth, compute_sky_mask
    except Exception as e:
        print(f"Warning: could not import sky utils for nested patch: {e}")
        return

    da3_core = getattr(model, "model", None)
    if da3_core is None:
        return

    orig_fn = getattr(da3_core, "_handle_sky_regions", None)
    if orig_fn is None:
        return

    def _patched(self, output, metric_output, sky_depth_def: float = 200.0):
        with torch.cuda.amp.autocast(enabled=False):
            non_sky_mask = compute_sky_mask(metric_output.sky.float(), threshold=0.3)
            non_sky_depth = output.depth.float()[non_sky_mask]
            if non_sky_depth.numel() > 100000:
                idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
                sampled_depth = non_sky_depth[idx]
            else:
                sampled_depth = non_sky_depth
            non_sky_max = min(torch.quantile(sampled_depth, 0.99), float(sky_depth_def))
        # Write back using dtype of depth to avoid dtype mismatch
        depth_dtype = output.depth.dtype
        conf_dtype = output.depth_conf.dtype
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth,
            output.depth_conf,
            non_sky_mask,
            max_depth=torch.tensor(non_sky_max, device=output.depth.device, dtype=depth_dtype),
        )
        # Ensure depth_conf dtype preserved
        output.depth_conf = output.depth_conf.to(conf_dtype)
        return output

    da3_core._handle_sky_regions = types.MethodType(_patched, da3_core)


def _register_model_input_cast_hook(root_model, target_dtype):
    # Cast all Tensor inputs to the root model to target_dtype. DepthAnything3 wraps the net in .model
    def hook(module, inputs):
        casted = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.dtype != target_dtype:
                casted.append(inp.to(target_dtype))
            elif isinstance(inp, (list, tuple)):
                casted.append(type(inp)(
                    x.to(target_dtype) if isinstance(x, torch.Tensor) and x.dtype != target_dtype else x
                    for x in inp
                ))
            else:
                casted.append(inp)
        return tuple(casted)
    root_model.register_forward_pre_hook(hook)


def _cast_model_params_and_buffers(model, dtype):
    # Force-cast all float params/buffers to the target dtype to avoid hidden fp32 leftovers
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    for b in model.buffers():
        if torch.is_floating_point(b):
            b.data = b.data.to(dtype)


def get_model(model_name, load_half=False):
    global model, current_model_name, current_model_load_half
    # Nested giant is sensitive to fp16 in dependency code; force fp32 to avoid dtype errors without modifying deps
    if model_name in NESTED_MODEL_NAMES and load_half:
        print("Notice: attempting fp16 with runtime patch for nested giant alignment.")
    if model is None or current_model_name != model_name or current_model_load_half != load_half:
        from depth_anything_3.api import DepthAnything3
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        display_VRAM_usage(f"before loading {model_name}")
        config_model_name = CONFIG_NAME_MAP.get(model_name, model_name)
        model = DepthAnything3(model_name=config_model_name)
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            from safetensors.torch import load_file
            # Keep weights on CPU during load to avoid fp32 GPU residency
            weight = load_file(model_path, device="cpu")
            if load_half:
                weight = {k: (v.half() if v.is_floating_point() else v) for k, v in weight.items()}
                model = model.half()  # set module params/buffers to half before loading
            model.load_state_dict(weight, strict=False)
        else:
            raise FileNotFoundError(f"Model file {model_name} not found. Please download it first.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        if load_half:
            # Convert any stragglers to fp16 and register input-cast hooks
            _convert_norm_layers(model, torch.float16)
            _cast_model_params_and_buffers(model, torch.float16)
            _register_norm_input_cast_hooks(model)
            _register_conv_input_cast_hooks(model)
            _register_linear_input_cast_hooks(model)
            # Cast inputs into the underlying network to fp16 to avoid float inputs with half weights
            _register_model_input_cast_hook(model, torch.float16)
            if hasattr(model, "model"):
                _register_model_input_cast_hook(model.model, torch.float16)
            if model_name in NESTED_MODEL_NAMES:
                _patch_nested_alignment_fp16(model)
                _patch_nested_sky_fp16(model)
        model.eval()
        # Debug: show where weights live and their dtypes to help diagnose overcommit/paging
        try:
            first_param = next(model.parameters())
            print(f"Model device: {first_param.device}, dtype: {first_param.dtype}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}; reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
            dtype_counts = _summarize_model_dtypes(model)
            print("Model dtype summary (dtype, device -> numel):", dict(dtype_counts))
        except StopIteration:
            print("Model has no parameters to summarize.")
        current_model_name = model_name
        current_model_load_half = load_half
        display_VRAM_usage(f"after loading {model_name}", include_peak=True)
    return model

def unload_current_model():
    global model, current_model_name, current_model_load_half
    if model is not None:
        display_VRAM_usage("before unload")
        # Drop references so PyTorch can free memory.
        # Avoid `del model` on the global name: other threads (e.g. UI poll)
        # may read `model` concurrently and would hit NameError.
        old_model = model
        model = None
        current_model_name = None
        current_model_load_half = None
        del old_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            display_VRAM_usage("after unload")

def run_segmentation(image_paths, conf_threshold=0.25, model_name="yolo11x-seg"):
    print(f"Loading {model_name} model...")
    display_VRAM_usage("before loading YOLO")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Please install it to use segmentation.")
        return None, None

    # Use selected model
    # model_name passed as argument
    model_path = get_any_model_path(f"{model_name}.pt")
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} to {model_path}...")
        url = _URLS.get(model_name, "")
        if not url:
            print(f"Error: No URL known for {model_name}. Please download {model_name}.pt manually to {os.path.dirname(model_path)}")
            return None, None
            
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.hub.download_url_to_file(url, model_path)
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            return None, None

    # Load model from specific path
    seg_model = YOLO(model_path) 
    display_VRAM_usage("after loading YOLO", include_peak=True)
    
    print(f"Running segmentation on {len(image_paths)} images...")
    
    # Run tracking
    # persist=True is important for video tracking
    # stream=True returns a generator, good for memory
    results = seg_model.track(source=image_paths, conf=conf_threshold, persist=True, stream=True, verbose=False)
    
    segmentation_data = []
    
    for i, r in enumerate(results):
        # r is a Results object
        # We need masks and track IDs
        frame_data = {
            "masks": [],
            "ids": [],
            "classes": [],
            "orig_shape": r.orig_shape
        }
        
        if r.masks is not None:
            # masks.data is a torch tensor of masks [N, H, W]
            masks = r.masks.data.cpu().numpy()
            
            # Crop masks to remove letterbox padding (YOLO pads to multiple of 32)
            # This ensures aspect ratio matches original image before we resize later
            h_orig, w_orig = r.orig_shape
            if len(masks.shape) == 3:
                _, h_mask, w_mask = masks.shape
                
                # Calculate scale factor that was used to fit image into mask
                scale = min(w_mask / w_orig, h_mask / h_orig)
                
                # Compute expected dimensions of the valid image area in the mask
                new_w = int(round(w_orig * scale))
                new_h = int(round(h_orig * scale))
                
                # Compute start offsets (centering)
                x_off = (w_mask - new_w) // 2
                y_off = (h_mask - new_h) // 2
                
                # Crop
                masks = masks[:, y_off : y_off + new_h, x_off : x_off + new_w]
            
            # Fix edge artifacts (sometimes edges are black)
            if len(masks.shape) == 3:
                for k in range(masks.shape[0]):
                    m = masks[k]
                    h_m, w_m = m.shape
                    
                    # Fix bottom edge
                    if h_m >= 3:
                        if np.max(m[-1, :]) == 0:
                            if np.max(m[-2, :]) == 0:
                                m[-2:, :] = m[-3, :]
                            else:
                                m[-1, :] = m[-2, :]

                    # Fix top edge
                    if h_m >= 3:
                        if np.max(m[0, :]) == 0:
                            if np.max(m[1, :]) == 0:
                                m[:2, :] = m[2, :]
                            else:
                                m[0, :] = m[1, :]

                    # Fix left edge
                    if w_m >= 3:
                        if np.max(m[:, 0]) == 0:
                            if np.max(m[:, 1]) == 0:
                                m[:, :2] = m[:, 2:3]
                            else:
                                m[:, 0] = m[:, 1]

            frame_data["masks"] = masks
            
            if r.boxes is not None and r.boxes.id is not None:
                frame_data["ids"] = r.boxes.id.int().cpu().numpy()
            else:
                # If no tracking IDs (e.g. first frame or lost track), use -1 or generate new ones?
                # If tracking is on, it should return IDs. If not, maybe just detection.
                # But we requested track().
                # If no ID, maybe it's a new object that wasn't tracked?
                # Let's use -1 for untracked
                if r.boxes is not None:
                    frame_data["ids"] = np.full(len(r.boxes), -1, dtype=int)
            
            if r.boxes is not None:
                frame_data["classes"] = r.boxes.cls.int().cpu().numpy()
                
        segmentation_data.append(frame_data)
        
        if i % 10 == 0:
            print(f"Segmented {i+1}/{len(image_paths)} images")

    display_VRAM_usage("after YOLO inference", include_peak=True)
    
    # Get class names
    class_names = seg_model.names

    # Cleanup
    del seg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        display_VRAM_usage("after unloading YOLO")
        
    return segmentation_data, class_names

class DownloadModelOperator(bpy.types.Operator):
    bl_idname = "da3.download_model"
    bl_label = "Download DA3 Model"

    # NEW: optional override for which model to download
    da3_override_model_name: bpy.props.StringProperty(
        name="Override Model Name",
        description="If set, download this model instead of the one selected in the scene",
        default="",
    )

    thread = None
    progress = 0
    old_progress = -1
    error_message = ""
    stop_event = None

    def invoke(self, context, event):
        model_name = self.da3_override_model_name or context.scene.da3_model_name
        self.model_path = get_model_path(model_name, context)
        
        if os.path.exists(self.model_path):
            self.report({'INFO'}, f"Model {model_name} already downloaded.")
            return {'FINISHED'}
        
        if model_name not in _URLS:
            self.report({'ERROR'}, f"Unknown model: {model_name}")
            return {'CANCELLED'}

        context.scene.da3_progress = 0
        self.progress = 0
        self.old_progress = -1
        self.error_message = ""
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self.download_file, args=(_URLS[model_name], self.model_path, context))
        self.thread.start()

        wm = context.window_manager
        # wm.progress_begin(0, 100)
        self.timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def download_file(self, url, path, context):
        try:
            print(f"Downloading model {url.split('/')[-2]}...")
            os.makedirs(os.path.dirname(path), exist_ok=True)

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

            if not self.stop_event.is_set():
                 self.progress = 100

        except Exception as e:
            self.error_message = f"Failed to download model: {e}"
        finally:
            self.stop_event.set()

    def modal(self, context, event):
        if event.type == 'TIMER':
            # context.window_manager.progress_update(self.progress)
            context.scene.da3_progress = self.progress
            if self.progress != self.old_progress:
                self.old_progress = self.progress
                # Force UI redraw
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

            if self.stop_event.is_set() and not self.thread.is_alive():
                wm = context.window_manager
                wm.event_timer_remove(self.timer)
                # wm.progress_end()
                context.scene.da3_progress = -1

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
            context.scene.da3_progress = -1

            wm = context.window_manager
            wm.event_timer_remove(self.timer)
            # wm.progress_end()

            self.report({'WARNING'}, "Download cancelled.")

            # Wait for the thread to finish before deleting the file
            if self.thread.is_alive():
                self.thread.join()

            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

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

    thread = None
    progress = 0
    old_progress = -1
    stage = ""
    old_stage = ""
    error_message = ""
    stop_event = None
    result_queue = None
    
    total_predicted_time = None
    start_time = None

    def start_progress_timer(self, total):
        self.start_time = time.time()
        self.total_predicted_time = total
        self.progress = 0
        self.stage = "Starting..."
        
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

    def update_progress_timer(self, expected_time, stage=""):
        if not self.total_predicted_time or self.total_predicted_time <= 0:
            print("Warning: total_predicted_time is zero or negative, cannot update progress.")
            return
        portion = expected_time / self.total_predicted_time * 100
        self.progress = portion
        self.stage = stage
        print(f"Progress: {stage}, {portion:.2f}%, elapsed: {time.time() - self.start_time:.2f}s")


    def invoke(self, context, event):
        self.input_folder = context.scene.da3_input_folder
        self.base_model_name = context.scene.da3_model_name
        self.use_metric = context.scene.da3_use_metric
        self.metric_mode = getattr(context.scene, "da3_metric_mode", "scale_base")
        self.use_ray_pose = getattr(context.scene, "da3_use_ray_pose", False)
        self.process_res = context.scene.da3_process_res
        self.process_res_method = context.scene.da3_process_res_method
        self.load_half_precision_model = getattr(context.scene, "da3_load_half_precision", False)
        # Use half-precision activations only when fp16 weights are requested
        self.use_half_precision = bool(self.load_half_precision_model)
        self.filter_edges = getattr(context.scene, "da3_filter_edges", True)
        self.min_confidence = getattr(context.scene, "da3_min_confidence", 0.5)
        self.output_debug_images = getattr(context.scene, "da3_output_debug_images", False)
        self.generate_mesh = getattr(context.scene, "da3_generate_mesh", False)
        self.batch_mode = getattr(context.scene, "da3_batch_mode", "skip_frames")
        self.batch_size = getattr(context.scene, "da3_batch_size", 10)
        self.frame_stride = getattr(context.scene, "da3_frame_stride", 1)
        self.ref_view_strategy = getattr(context.scene, "da3_ref_view_strategy", "saddle_balanced")
        self.streaming_loop_enable = getattr(context.scene, "da3_streaming_loop_enable", True)
        self.streaming_use_db_ow = getattr(context.scene, "da3_streaming_use_db_ow", False)
        self.streaming_align_lib = getattr(context.scene, "da3_streaming_align_lib", "torch")
        self.streaming_align_method = getattr(context.scene, "da3_streaming_align_method", "sim3")
        self.streaming_depth_threshold = getattr(context.scene, "da3_streaming_depth_threshold", 15.0)
        self.streaming_save_debug = getattr(context.scene, "da3_streaming_save_debug", False)
        self.streaming_conf_threshold_coef = getattr(context.scene, "da3_streaming_conf_threshold_coef", 0.75)
        self.use_segmentation = getattr(context.scene, "da3_use_segmentation", False)
        self.segmentation_conf = getattr(context.scene, "da3_segmentation_conf", 0.25)
        self.segmentation_model = getattr(context.scene, "da3_segmentation_model", "yolo11x-seg")
        self.detect_motion = getattr(context.scene, "da3_detect_motion", False)
        self.motion_threshold = getattr(context.scene, "da3_motion_threshold", 0.1)
        
        # Prime the model folder cache in the main thread
        get_configured_model_folder(context)
        
        if self.process_res % 14 != 0:
            self.report({'ERROR'}, "Process resolution must be a multiple of 14.")
            return {'CANCELLED'}
        
        if not self.input_folder or not os.path.isdir(self.input_folder):
            self.report({'ERROR'}, "Please select a valid input folder.")
            return {'CANCELLED'}

        context.scene.da3_progress = 0
        self.progress = 0
        self.error_message = ""
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue()

        self.thread = threading.Thread(target=self.generate_worker, args=(context,))
        self.thread.start()

        wm = context.window_manager
        self.timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def handle_batch_result(self, prediction, batch_indices, batch_number, folder_name, all_segmentation_data, segmentation_class_names):
        # Detect motion for this batch only if enabled (suboptimal but better than nothing?)
        # Or just skip motion detection for incremental updates?
        # The user didn't specify, but "results as soon as ready" implies we don't wait.
        if self.detect_motion:
             compute_motion_scores([prediction], threshold_ratio=self.motion_threshold)

        # Extract segmentation data for this batch
        batch_segmentation = None
        if all_segmentation_data:
            batch_segmentation = [all_segmentation_data[j] for j in batch_indices]
        
        batch_paths = [self.image_paths[j] for j in batch_indices]

        combined_predictions = convert_prediction_to_dict(
            prediction, 
            batch_paths, 
            output_debug_images=self.output_debug_images,
            segmentation_data=batch_segmentation,
            class_names=segmentation_class_names
        )
        
        # Create batch collection
        self.result_queue.put({
            "type": "BATCH_READY",
            "data": combined_predictions,
            "batch_number": batch_number,
            "folder_name": folder_name,
            "batch_indices": batch_indices,
            "generate_mesh": self.generate_mesh,
            "filter_edges": self.filter_edges,
            "min_confidence": self.min_confidence
        })

    def _write_streaming_config(self, model_path, chunk_size, overlap, loop_chunk_size, output_dir, ref_view_strategy):
        # Build a minimal YAML config string using current UI settings
        weights_cfg = model_path.replace('\\', '/')
        config_json = os.path.join(os.path.dirname(model_path), "config.json").replace('\\', '/')
        salad_path = (STREAMING_DIR / "weights" / "dino_salad.ckpt").as_posix()

        yaml_str = f"""Weights:
  DA3: '{weights_cfg}'
  DA3_CONFIG: '{config_json}'
  SALAD: '{salad_path}'

Model:
  chunk_size: {chunk_size}
  overlap: {overlap}
  loop_chunk_size: {loop_chunk_size} # imgs of loop chunk = 2 * loop_chunk_size
  loop_enable: {str(self.streaming_loop_enable).lower()}
  useDBoW: {str(self.streaming_use_db_ow).lower()}
  delete_temp_files: True
  align_lib: '{self.streaming_align_lib}'
  align_method: '{self.streaming_align_method}'
  scale_compute_method: 'auto'
  align_type: 'dense'

  ref_view_strategy: '{ref_view_strategy}'
  ref_view_strategy_loop: '{ref_view_strategy}'
  depth_threshold: {self.streaming_depth_threshold}

  save_depth_conf_result: True
  save_debug_info: {str(self.streaming_save_debug).lower()}

  Sparse_Align:
    keypoint_select: 'orb'
    keypoint_num: 5000

  IRLS:
    delta: 0.1
    max_iters: 5
    tol: 1e-9

  Pointcloud_Save:
        sample_ratio: 1.0
    conf_threshold_coef: 0.75

Loop:
  SALAD:
    image_size: [336, 336]
    batch_size: 32
    similarity_threshold: 0.85
    top_k: 5
    use_nms: True
    nms_threshold: 25

  SIM3_Optimizer:
    lang_version: 'cpp'
    max_iterations: 30
    lambda_init: 1e-6
"""
        fd, cfg_path = tempfile.mkstemp(prefix="da3_streaming_", suffix=".yaml", dir=output_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_str)
        return cfg_path

    def _run_streaming_pipeline(self, context):
        folder_name = os.path.basename(os.path.normpath(self.input_folder))
        output_dir = getattr(context.scene, "da3_streaming_output", "") or os.path.join(self.input_folder, "da3_streaming_output")
        os.makedirs(output_dir, exist_ok=True)

        model_path = get_model_path(self.base_model_name, context)
        if not os.path.exists(model_path):
            self.result_queue.put({"type": "ERROR", "message": f"Streaming model not found: {model_path}"})
            return

        chunk_size = max(1, int(self.batch_size))
        overlap = max(1, chunk_size // 2)
        loop_chunk_size = overlap

        cfg_path = self._write_streaming_config(model_path, chunk_size, overlap, loop_chunk_size, output_dir, self.ref_view_strategy)

        env = os.environ.copy()
        py_path_parts = []
        existing = env.get("PYTHONPATH", "")
        if existing:
            py_path_parts.append(existing)
        py_path_parts.insert(0, os.fspath(DEPS_DA3_PATH))
        py_path_parts.insert(0, os.fspath(DEPS_PUBLIC_PATH))
        env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

        cmd = [
            sys.executable,
            os.fspath(STREAMING_SCRIPT),
            "--image_dir", os.fspath(self.input_folder),
            "--config", os.fspath(cfg_path),
            "--output_dir", os.fspath(output_dir),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=os.fspath(STREAMING_DIR),
                env=env,
                text=True,
                capture_output=True,
            )
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)
            if proc.returncode != 0:
                self.result_queue.put({"type": "ERROR", "message": f"DA3-Streaming failed (code {proc.returncode})."})
                return
        except Exception as e:
            self.result_queue.put({"type": "ERROR", "message": f"Failed to run DA3-Streaming: {e}"})
            return

        combined_pcd = os.path.join(output_dir, "combined_pcd.ply")
        pcd_dir = os.path.join(output_dir, "pcd")
        if not os.path.exists(combined_pcd):
            self.result_queue.put({"type": "ERROR", "message": "Streaming completed but combined_pcd.ply not found."})
            return

        self.result_queue.put({"type": "INIT_COLLECTION", "folder_name": folder_name})
        # Provide both combined PLY and the directory containing per-chunk PLYs
        self.result_queue.put({"type": "STREAMING_PLY", "path": combined_pcd, "pcd_dir": pcd_dir, "folder_name": folder_name})
        self.result_queue.put({"type": "DONE"})

    def _import_streaming_ply(self, context, msg):
        try:
            import numpy as _np
            from plyfile import PlyData

            folder_name = msg["folder_name"]
            ply_path = msg["path"]

            # Optionally import per-chunk PLYs into separate collections
            pcd_dir = msg.get("pcd_dir")
            use_chunks = getattr(context.scene, "da3_streaming_chunk_collections", False) and pcd_dir and os.path.isdir(pcd_dir)

            # If segmentation is enabled, import streaming results via the same
            # code path as other batch modes (import_point_cloud/import_mesh_from_depth)
            use_segmentation = bool(getattr(context.scene, "da3_use_segmentation", False))
            generate_mesh = bool(getattr(context.scene, "da3_generate_mesh", False))
            filter_edges = bool(getattr(context.scene, "da3_filter_edges", True))
            min_confidence = float(getattr(context.scene, "da3_min_confidence", 0.5))

            def _try_import_streaming_cameras(_output_dir, _collection):
                if not _output_dir or _collection is None:
                    return False
                poses_path = os.path.join(_output_dir, "camera_poses.txt")
                intrinsics_path = os.path.join(_output_dir, "intrinsic.txt")
                if not (os.path.exists(poses_path) and os.path.exists(intrinsics_path)):
                    return False

                intrinsics = []
                extrinsics = []
                with open(poses_path, "r") as f:
                    for line in f:
                        vals = [float(x) for x in line.strip().split() if x.strip()]
                        if len(vals) != 16:
                            continue
                        c2w = _np.array(vals, dtype=_np.float64).reshape((4, 4))
                        # The streaming pipeline saves camera poses as C2W (camera->world).
                        # `create_cameras` expects W2C (world->camera), so invert.
                        try:
                            w2c = _np.linalg.inv(c2w)
                        except Exception:
                            continue
                        extrinsics.append(w2c[:3, :4].astype(_np.float32))

                with open(intrinsics_path, "r") as f:
                    for line in f:
                        parts = [p for p in line.strip().split() if p.strip()]
                        if len(parts) < 4:
                            continue
                        fx, fy, cx, cy = [float(x) for x in parts[:4]]
                        K = _np.array(
                            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                            dtype=_np.float32,
                        )
                        intrinsics.append(K)

                if len(intrinsics) != len(extrinsics) or len(intrinsics) == 0:
                    return False

                preds = {"intrinsic": intrinsics, "extrinsic": extrinsics}
                image_paths = getattr(self, "image_paths", None)
                # DA3/streaming intrinsics are expressed in the *processed* image
                # coordinate system (e.g. 504x280), not the original file resolution.
                # Require the persisted processed image size; error if missing.
                size_path = os.path.join(_output_dir, "intrinsic_image_size.txt")
                if not os.path.exists(size_path):
                    raise RuntimeError("Missing intrinsic_image_size.txt: cannot determine processed image size for streaming camera import. Please re-run streaming with a recent version.")
                with open(size_path, "r") as f:
                    parts = [p for p in f.readline().strip().split() if p.strip()]
                if len(parts) < 2:
                    raise RuntimeError(f"Malformed intrinsic_image_size.txt: expected 'W H' on first line, got: {parts}")
                iw = int(float(parts[0]))
                ih = int(float(parts[1]))
                if iw <= 0 or ih <= 0:
                    raise RuntimeError(f"Invalid image size in intrinsic_image_size.txt: W={iw}, H={ih}")
                image_width = iw
                image_height = ih

                if image_paths:
                    preds["image_paths"] = image_paths

                create_cameras(preds, collection=_collection, image_width=image_width, image_height=image_height)
                return True

            # Unified parity path: segmentation, mesh, or motion (non-chunk import)
            if (use_segmentation or self.detect_motion or generate_mesh) and not use_chunks:
                output_dir = os.path.dirname(ply_path)
                seg_dir = os.path.join(output_dir, "segmented_chunks")
                if os.path.isdir(seg_dir):
                    chunk_files = []
                    for f in os.listdir(seg_dir):
                        if not (f.startswith("chunk_") and f.lower().endswith(".npy")):
                            continue
                        if f.endswith("_segmaps.npy"):
                            continue
                        try:
                            idx = int(f[len("chunk_") : -len(".npy")])
                        except Exception:
                            continue
                        chunk_files.append((idx, f))

                    if chunk_files:
                        chunk_files.sort(key=lambda t: t[0])
                        points_list = []
                        images_list = []
                        conf_list = []
                        seg_id_list = []
                        id_to_class = {}
                        class_names = {}
                        motion_scores_list = []

                        for _, fname in chunk_files:
                            seg_chunk_path = os.path.join(seg_dir, fname)
                            seg_chunk_obj = _np.load(seg_chunk_path, allow_pickle=True)
                            seg_chunk = seg_chunk_obj.item() if hasattr(seg_chunk_obj, 'item') else seg_chunk_obj
                            if not isinstance(seg_chunk, dict):
                                continue

                            points = seg_chunk.get("world_points")
                            images_u8 = seg_chunk.get("images")
                            conf = seg_chunk.get("conf")
                            if points is None or images_u8 is None or conf is None:
                                continue
                            conf += 1.0

                            points_list.append(points)
                            images_list.append(images_u8.astype(_np.float32) / 255.0)
                            conf_list.append(conf)

                            seg_id_map = seg_chunk.get("seg_id_map")
                            if seg_id_map is not None:
                                seg_id_list.append(seg_id_map)

                            # Merge maps conservatively
                            for k, v in (seg_chunk.get("id_to_class") or {}).items():
                                if k not in id_to_class:
                                    id_to_class[k] = v
                            if not class_names:
                                class_names = seg_chunk.get("class_names") or {}

                            # Motion scores (if present)
                            if "motion_scores" in seg_chunk:
                                motion_scores_list.append(seg_chunk["motion_scores"])

                        if points_list and images_list and conf_list:
                            points_all = _np.concatenate(points_list, axis=0)
                            images_all = _np.concatenate(images_list, axis=0)
                            conf_all = _np.concatenate(conf_list, axis=0)

                            d = {
                                "world_points_from_depth": points_all,
                                "images": images_all,
                                "conf": conf_all,
                            }
                            if seg_id_list:
                                try:
                                    d["seg_id_map"] = _np.concatenate(seg_id_list, axis=0)
                                except Exception:
                                    pass
                                d["id_to_class"] = id_to_class
                                d["class_names"] = class_names
                            if motion_scores_list:
                                try:
                                    d["motion_scores"] = _np.concatenate(motion_scores_list, axis=0)
                                except Exception:
                                    pass

                            target_col = self.parent_col
                            if not target_col:
                                target_col = bpy.data.collections.new(folder_name)
                                context.scene.collection.children.link(target_col)

                            # Detect motion if enabled
                            if self.detect_motion:
                                from types import SimpleNamespace
                                d_obj = SimpleNamespace(
                                    depth=d.get("conf"),
                                    conf=d.get("conf"),
                                    world_points_from_depth=d.get("world_points_from_depth"),
                                    images=d.get("images"),
                                    seg_id_map=d.get("seg_id_map"),
                                    id_to_class=d.get("id_to_class"),
                                    class_names=d.get("class_names"),
                                    motion_scores=d.get("motion_scores"),
                                )
                                compute_motion_scores([d_obj], threshold_ratio=self.motion_threshold)

                            # Import as mesh or point cloud
                            if generate_mesh:
                                import_mesh_from_depth(d, collection=target_col, filter_edges=filter_edges, min_confidence=min_confidence)
                            else:
                                import_point_cloud(d, collection=target_col, filter_edges=filter_edges, min_confidence=min_confidence)
                            try:
                                _try_import_streaming_cameras(output_dir, target_col)
                            except Exception:
                                pass
                            return {'FINISHED'}

            if use_chunks:
                # Parent collection for this streaming run
                target_parent = self.parent_col
                if not target_parent:
                    target_parent = bpy.data.collections.new(folder_name)
                    context.scene.collection.children.link(target_parent)

                # Load all .ply files in pcd_dir sorted by name
                chunk_files = sorted([f for f in os.listdir(pcd_dir) if f.lower().endswith('.ply')])
                for idx, fname in enumerate(chunk_files):
                    chunk_path = os.path.join(pcd_dir, fname)
                    try:
                        # Segmentation-enabled: load aligned chunk arrays and import like other batch modes
                        if use_segmentation:
                            output_dir = os.path.dirname(pcd_dir)
                            seg_dir = os.path.join(output_dir, "segmented_chunks")
                            seg_chunk_path = os.path.join(seg_dir, f"chunk_{idx}.npy")

                            if os.path.exists(seg_chunk_path):
                                seg_chunk_obj = _np.load(seg_chunk_path, allow_pickle=True)
                                seg_chunk = seg_chunk_obj.item() if hasattr(seg_chunk_obj, 'item') else seg_chunk_obj

                                # Expect dict with world_points/conf/images; may also include segmaps
                                points = seg_chunk.get("world_points")
                                images_u8 = seg_chunk.get("images")
                                conf = seg_chunk.get("conf")
                                if points is None or images_u8 is None or conf is None:
                                    raise ValueError("Segmented chunk data missing required keys")
                                conf += 1.0

                                images = images_u8.astype(_np.float32) / 255.0

                                d = {
                                    "world_points_from_depth": points,
                                    "images": images,
                                    "conf": conf,
                                }

                                if "seg_id_map" in seg_chunk and seg_chunk["seg_id_map"] is not None:
                                    d["seg_id_map"] = seg_chunk["seg_id_map"]
                                    d["id_to_class"] = seg_chunk.get("id_to_class", {}) or {}
                                    d["class_names"] = seg_chunk.get("class_names", {}) or {}

                                chunk_col_name = f"{folder_name}_chunk_{idx}"
                                chunk_col = bpy.data.collections.new(chunk_col_name)
                                target_parent.children.link(chunk_col)

                                if generate_mesh:
                                    import_mesh_from_depth(d, collection=chunk_col, filter_edges=filter_edges, min_confidence=min_confidence)
                                else:
                                    import_point_cloud(d, collection=chunk_col, filter_edges=filter_edges, min_confidence=min_confidence)

                                continue

                        plydata = PlyData.read(chunk_path)
                        vertices = plydata["vertex"].data
                        pts = _np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(_np.float32)
                        pts[:, [0, 1, 2]] = pts[:, [0, 2, 1]]
                        pts[:, 2] = -pts[:, 2]

                        if {"red", "green", "blue"}.issubset(vertices.dtype.names):
                            cols = _np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(_np.float32) / 255.0
                        else:
                            cols = _np.ones((len(pts), 3), dtype=_np.float32)
                        cols = _np.hstack((cols, _np.ones((len(pts), 1), dtype=_np.float32)))

                        if "confidence" in vertices.dtype.names:
                            confs = vertices["confidence"].astype(_np.float32)
                            confs += 1.0
                        else:
                            confs = _np.ones((len(pts),), dtype=_np.float32)

                        chunk_col_name = f"{folder_name}_chunk_{idx}"
                        chunk_col = bpy.data.collections.new(chunk_col_name)
                        target_parent.children.link(chunk_col)

                        obj_name = f"{chunk_col_name}_Cloud"
                        create_point_cloud_object(obj_name, pts, cols, confs, collection=chunk_col)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        continue
                try:
                    output_dir = msg.get("output_dir") if msg.get("output_dir") else (os.path.dirname(pcd_dir) if pcd_dir else None)
                    _try_import_streaming_cameras(output_dir, target_parent)
                except Exception:
                    import traceback
                    traceback.print_exc()
                return {'FINISHED'}

            # Fallback: import combined PLY into single collection
            target_col = self.parent_col
            if not target_col:
                target_col = bpy.data.collections.new(folder_name)
                context.scene.collection.children.link(target_col)

            plydata = PlyData.read(ply_path)
            vertices = plydata["vertex"].data  # structured numpy array
            pts = _np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(_np.float32)
            # Match the same coordinate conversion used in import_point_cloud
            pts[:, [0, 1, 2]] = pts[:, [0, 2, 1]]
            pts[:, 2] = -pts[:, 2]

            if {"red", "green", "blue"}.issubset(vertices.dtype.names):
                cols = _np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(_np.float32) / 255.0
            else:
                cols = _np.ones((len(pts), 3), dtype=_np.float32)
            cols = _np.hstack((cols, _np.ones((len(pts), 1), dtype=_np.float32)))

            if "confidence" in vertices.dtype.names:
                confs = vertices["confidence"].astype(_np.float32)
                confs += 1.0                            
                print(f"DEBUG: Loaded confidence from PLY - min: {confs.min():.4f}, max: {confs.max():.4f}, mean: {confs.mean():.4f}")
            else:
                confs = _np.ones((len(pts),), dtype=_np.float32)
                print("DEBUG: No confidence in PLY, using all 1.0")
            obj_name = f"{folder_name}_StreamingCloud"
            create_point_cloud_object(obj_name, pts, cols, confs, collection=target_col)
            try:
                _try_import_streaming_cameras(os.path.dirname(ply_path), target_col)
            except Exception:
                pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Failed to import streaming point cloud: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    def generate_worker(self, context):
        try:
            # Get image paths
            import glob
            self.image_paths = sorted(glob.glob(os.path.join(self.input_folder, "*.[jJpP][pPnN][gG]")))
            if not self.image_paths:
                self.result_queue.put({"type": "ERROR", "message": "No images found in the input folder."})
                return
            
            print(f"Total images: {len(self.image_paths)}")

            # Apply frame stride subsampling
            frame_stride = getattr(context.scene, "da3_frame_stride", 1)
            if frame_stride > 1:
                self.image_paths = self.image_paths[::frame_stride]
                print(f"After frame stride {frame_stride}: {len(self.image_paths)} images")

            if self.batch_mode == "skip_frames" and len(self.image_paths) > self.batch_size:
                import numpy as np
                indices = np.linspace(0, len(self.image_paths) - 1, self.batch_size, dtype=int)
                self.image_paths = [self.image_paths[i] for i in indices]
            
            print(f"Processing {len(self.image_paths)} images...")

            # Initialize progress bar
            LoadModelTime = 9.2 # seconds
            AlignBatchesTime = 0.29
            AddImagePointsTime = 0.27
            BatchTimePerImage = 4.9 # it's actually quadratic but close enough
            MetricLoadModelTime = 19.25
            MetricBatchTimePerImage = 0.62
            MetricCombineTime = 0.12
            if current_model_name == self.base_model_name:
                LoadModelTime = 0
            needs_alignment = self.batch_mode in ("last_frame_overlap", "first_last_overlap")
            BaseTimeEstimate = LoadModelTime + BatchTimePerImage * len(self.image_paths)
            if needs_alignment:
                BaseTimeEstimate += AlignBatchesTime
            if self.use_metric:
                MetricTimeEstimate = BaseTimeEstimate + MetricLoadModelTime
                if self.metric_mode == "scale_base":
                    MetricTimeEstimate += MetricBatchTimePerImage * self.batch_size
                else:
                    MetricTimeEstimate += MetricBatchTimePerImage * len(self.image_paths)
                AfterCombineTimeEstimate = MetricTimeEstimate
                if needs_alignment:
                    AfterCombineTimeEstimate += AlignBatchesTime
                AfterCombineTimeEstimate += MetricCombineTime
            else:
                MetricTimeEstimate = BaseTimeEstimate
                AfterCombineTimeEstimate = BaseTimeEstimate
                if needs_alignment:
                    AfterCombineTimeEstimate += AlignBatchesTime
            TotalTimeEstimate = AfterCombineTimeEstimate + AddImagePointsTime*len(self.image_paths)
            self.start_progress_timer(TotalTimeEstimate)

            def progress_callback(progress_value, progress_msg=None):
                self.progress = progress_value
                if progress_msg:
                    self.stage = progress_msg
                if self.stop_event.is_set():
                    return True
                return False

            # 0) Run Segmentation
            all_segmentation_data = None
            segmentation_class_names = None
            if self.use_segmentation:
                print("Running segmentation...")
                # Ensure DA3 model is unloaded
                unload_current_model()
                all_segmentation_data, segmentation_class_names = run_segmentation(
                    self.image_paths, 
                    conf_threshold=self.segmentation_conf, 
                    model_name=self.segmentation_model
                )
                if all_segmentation_data is None:
                    print("Segmentation failed or cancelled. Proceeding without segmentation.")

            # Prepare for import
            folder_name = os.path.basename(os.path.normpath(self.input_folder))
            self.result_queue.put({"type": "INIT_COLLECTION", "folder_name": folder_name})

            # 1) run base model (or streaming pipeline)
            if self.batch_mode == "da3_streaming":
                try:
                    metric_first_chunk_prediction = None
                    if self.use_metric:
                        metric_model_name = "da3metric-large"
                        metric_path = get_model_path(metric_model_name, context)
                        if not os.path.exists(metric_path):
                            print("Metric model not downloaded; streaming will run without metric scale.")
                        else:
                            # Run the metric model FIRST (before we ever load the base model),
                            # and pass its first-chunk prediction into the streaming runner.
                            # The streaming pipeline will compute scale after the first base chunk
                            # without re-running that base chunk.
                            sample_n = max(1, int(self.batch_size))
                            sample_paths = self.image_paths[: min(sample_n, len(self.image_paths))]

                            print("Running metric model on first chunk to compute scale later...")
                            unload_current_model()
                            metric_model = get_model(metric_model_name, load_half=self.load_half_precision_model)
                            with torch.no_grad():
                                metric_first_chunk_prediction = metric_model.inference(
                                    sample_paths,
                                    ref_view_strategy=self.ref_view_strategy,
                                )
                            metric_model = None
                            unload_current_model()

                    self.update_progress_timer(0, f"Loading {self.base_model_name} model...")
                    base_model = get_model(self.base_model_name, load_half=self.load_half_precision_model)
                    self.update_progress_timer(LoadModelTime, "Loaded base model")

                    res = streaming_runner.run_streaming(
                        image_dir=self.input_folder,
                        image_paths=self.image_paths,
                        output_dir=getattr(context.scene, "da3_streaming_output", "")
                        or os.path.join(self.input_folder, "da3_streaming_output"),
                        model_path=get_model_path(self.base_model_name, context),
                        chunk_size=self.batch_size,
                        overlap=max(1, self.batch_size // 2),
                        model=base_model,
                        progress_callback=progress_callback,
                        ref_view_strategy=self.ref_view_strategy,
                        loop_enable=self.streaming_loop_enable,
                        use_db_ow=self.streaming_use_db_ow,
                        align_lib=self.streaming_align_lib,
                        align_method=self.streaming_align_method,
                        depth_threshold=self.streaming_depth_threshold,
                        save_debug=self.streaming_save_debug,
                        conf_threshold_coef=self.streaming_conf_threshold_coef,
                        filter_edges=self.filter_edges,
                        segmentation_data=all_segmentation_data,
                        segmentation_class_names=segmentation_class_names,
                        metric_first_chunk_prediction=metric_first_chunk_prediction,
                    )
                    # Return both the combined PLY and the per-chunk PCD directory
                    self.result_queue.put({
                        "type": "STREAMING_PLY",
                        "path": res["combined_ply"],
                        "pcd_dir": res.get("pcd_dir"),
                        "folder_name": folder_name,
                    })
                    self.result_queue.put({"type": "DONE"})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.result_queue.put({"type": "ERROR", "message": f"Streaming failed: {e}"})
                return

            self.update_progress_timer(0, f"Loading {self.base_model_name} model...")
            base_model = get_model(self.base_model_name, load_half=self.load_half_precision_model)
            self.update_progress_timer(LoadModelTime, "Loaded base model")
            print("Running base model inference...")
            
            all_base_predictions = []
            prev_pred = None
            prev_indices = None
            batch_counter = 0
            
            # Helper to process result immediately if not using metric
            def process_result_if_ready(prediction, indices):
                nonlocal prev_pred, prev_indices, batch_counter
                if not self.use_metric:
                    # Align if needed
                    if prev_pred is not None and self.batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                        prediction = align_single_batch(prediction, indices, prev_pred, prev_indices)
                    
                    prev_pred = prediction
                    prev_indices = indices
                    
                    self.handle_batch_result(prediction, indices, batch_counter, folder_name, all_segmentation_data, segmentation_class_names)
                    batch_counter += 1
                else:
                    all_base_predictions.append((prediction, indices))
            
            if self.batch_mode == "no_overlap":
                num_batches = (len(self.image_paths) + self.batch_size - 1) // self.batch_size
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(self.image_paths))
                    batch_paths = self.image_paths[start_idx:end_idx]
                    batch_indices = list(range(start_idx, end_idx))
                    print(f"Batch {batch_idx + 1}/{num_batches}:")
                    prediction = run_model(batch_paths, base_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                    self.update_progress_timer(LoadModelTime + end_idx * BatchTimePerImage, f"Base batch {batch_idx + 1}")
                    process_result_if_ready(prediction, batch_indices)

            elif self.batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                if self.batch_mode == "last_frame_overlap":
                    step = self.batch_size - 1
                    num_batches = (len(self.image_paths) + step - 1) // step  # Ceiling division
                    for batch_idx, start_idx in enumerate(range(0, len(self.image_paths), step)):
                        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
                        batch_paths = self.image_paths[start_idx:end_idx]
                        batch_indices = list(range(start_idx, end_idx))
                        print(f"Batch {batch_idx + 1}/{num_batches}:")
                        prediction = run_model(batch_paths, base_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                        self.update_progress_timer(LoadModelTime + end_idx * BatchTimePerImage, f"Base batch {batch_idx + 1}")
                        process_result_if_ready(prediction, batch_indices)
                else:
                    # New scheme: (0..9) (0, 9, 10..17) (10, 17, 18..25)
                    N = len(self.image_paths)
                    if self.batch_size < 3:
                        step = 1
                    else:
                        step = self.batch_size - 2

                    # First batch
                    start = 0
                    end = min(self.batch_size, N)
                    batch_indices = list(range(start, end))
                    current_new_indices = batch_indices
                    
                    remaining_start = end
                    
                    if step > 0:
                        num_batches = 1 + max(0, (N - end + step - 1) // step)
                    else:
                        num_batches = (N + self.batch_size - 1) // self.batch_size

                    batch_idx = 0
                    while True:
                        batch_paths = [self.image_paths[i] for i in batch_indices]
                        print(f"Batch {batch_idx + 1}/{num_batches}:")
                        prediction = run_model(batch_paths, base_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                        end_idx = batch_indices[-1] + 1
                        self.update_progress_timer(LoadModelTime + end_idx * BatchTimePerImage, f"Base batch {batch_idx + 1}")
                        process_result_if_ready(prediction, batch_indices.copy())

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
                prediction = run_model(self.image_paths, base_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                self.update_progress_timer(LoadModelTime + len(self.image_paths) * BatchTimePerImage, "Base batch complete")
                process_result_if_ready(prediction, list(range(len(self.image_paths))))
            
            self.update_progress_timer(BaseTimeEstimate, "Base inference complete")

            # If NOT using metric, we are done!
            if not self.use_metric:
                unload_if_overcommitted("post-base-inference")
                self.result_queue.put({"type": "DONE"})
                return

            # 2) if metric enabled and weights available:
            all_metric_predictions = []
            metric_available = False
            
            if self.use_metric:
                metric_path = get_model_path("da3metric-large")
                if os.path.exists(metric_path):
                    metric_available = True
                    # free base model from VRAM before loading metric
                    print("Unloading base model and loading metric model...")
                    base_model = None
                    unload_current_model()

                    metric_model = get_model("da3metric-large", load_half=self.load_half_precision_model)
                    self.update_progress_timer(BaseTimeEstimate + MetricLoadModelTime, "Loaded metric model")
                    print("Running metric model inference...")
                    
                    if self.metric_mode == "scale_base":
                        # In scale_base mode, run **one** metric batch over all images.
                        N = len(self.image_paths)
                        start = 0
                        end = min(self.batch_size, N)
                        batch_indices = list(range(start, end))
                        batch_paths = [self.image_paths[i] for i in batch_indices]
                        prediction = run_model(
                            batch_paths,
                            metric_model,
                            self.process_res,
                            self.process_res_method,
                            use_half=self.use_half_precision,
                            use_ray_pose=self.use_ray_pose,
                            ref_view_strategy=self.ref_view_strategy
                        )
                        self.update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + end * MetricBatchTimePerImage, "Metric batch complete")
                        all_metric_predictions.append((prediction, batch_indices.copy()))
                    else:
                        # For other metric modes, keep previous batching behaviour
                        if self.batch_mode == "no_overlap":
                            num_batches = (len(self.image_paths) + self.batch_size - 1) // self.batch_size
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * self.batch_size
                                end_idx = min(start_idx + self.batch_size, len(self.image_paths))
                                batch_paths = self.image_paths[start_idx:end_idx]
                                batch_indices = list(range(start_idx, end_idx))
                                print(f"Batch {batch_idx + 1}/{num_batches}:")
                                prediction = run_model(batch_paths, metric_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                                self.update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + end_idx * MetricBatchTimePerImage, f"Metric batch {batch_idx + 1}")
                                all_metric_predictions.append((prediction, batch_indices))

                        elif self.batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                            # Process in overlapping batches for metric too (mirror base logic)
                            if self.batch_mode == "last_frame_overlap":
                                step = self.batch_size - 1
                                num_batches = (len(self.image_paths) + step - 1) // step
                                for batch_idx, start_idx in enumerate(range(0, len(self.image_paths), step)):
                                    end_idx = min(start_idx + self.batch_size, len(self.image_paths))
                                    batch_paths = self.image_paths[start_idx:end_idx]
                                    batch_indices = list(range(start_idx, end_idx))
                                    print(f"Batch {batch_idx + 1}/{num_batches}:")
                                    prediction = run_model(batch_paths, metric_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                                    self.update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + end_idx * MetricBatchTimePerImage, f"Metric batch {batch_idx + 1}")
                                    all_metric_predictions.append((prediction, batch_indices))
                            else:
                                N = len(self.image_paths)
                                if self.batch_size < 3:
                                    step = 1
                                else:
                                    step = self.batch_size - 2

                                start = 0
                                end = min(self.batch_size, N)
                                batch_indices = list(range(start, end))
                                current_new_indices = batch_indices
                                
                                remaining_start = end
                                
                                if step > 0:
                                    num_batches = 1 + max(0, (N - end + step - 1) // step)
                                else:
                                    num_batches = (N + self.batch_size - 1) // self.batch_size

                                batch_idx = 0
                                while True:
                                    batch_paths = [self.image_paths[i] for i in batch_indices]
                                    print(f"Batch {batch_idx + 1}/{num_batches}:")
                                    prediction = run_model(batch_paths, metric_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                                    end_idx = batch_indices[-1] + 1
                                    self.update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + end_idx * MetricBatchTimePerImage, f"Metric batch {batch_idx + 1}")
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
                            prediction = run_model(self.image_paths, metric_model, self.process_res, self.process_res_method, use_half=self.use_half_precision, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy)
                            all_metric_predictions.append((prediction, list(range(len(self.image_paths)))))
                            self.update_progress_timer(BaseTimeEstimate + MetricLoadModelTime + len(self.image_paths) * MetricBatchTimePerImage, "Metric batch complete")
                    metric_model = None
                    unload_current_model()
                else:
                    print("Metric model not downloaded; using non-metric depth only.")
            
            
            self.update_progress_timer(MetricTimeEstimate, "Metric inference complete")
            # Align base batches. Metric is **not** aligned in scale_base mode.
            if self.batch_mode in {"last_frame_overlap", "first_last_overlap"}:
                aligned_base_predictions = align_batches(all_base_predictions)
                if metric_available:
                    aligned_metric_predictions = [p[0] for p in all_metric_predictions]
            else:
                aligned_base_predictions = [p[0] for p in all_base_predictions]
                if metric_available:
                    aligned_metric_predictions = [p[0] for p in all_metric_predictions]
            self.update_progress_timer(MetricTimeEstimate + AlignBatchesTime, "Align batches complete")

            # Creating the main collection named after the folder has been moved
            # Combine the base and metric predictions
            if metric_available:
                all_combined_predictions = combine_base_and_metric(aligned_base_predictions, aligned_metric_predictions)
            else:
                all_combined_predictions = aligned_base_predictions
            self.update_progress_timer(AfterCombineTimeEstimate, "Combined predictions complete")

            # Detect motion
            if self.detect_motion:
                self.update_progress_timer(AfterCombineTimeEstimate + 1.0, "Detecting motion...")
                compute_motion_scores(all_combined_predictions, threshold_ratio=self.motion_threshold)
                self.update_progress_timer(AfterCombineTimeEstimate + 1.0, "Motion detection complete")

            # Prepare for import
            # folder_name = os.path.basename(os.path.normpath(self.input_folder))
            # self.result_queue.put({"type": "INIT_COLLECTION", "folder_name": folder_name})

            for batch_number, batch_prediction in enumerate(all_combined_predictions):
                batch_indices = all_base_predictions[batch_number][1]
                # batch_paths = [self.image_paths[j] for j in batch_indices]
                
                # Extract segmentation data for this batch
                # batch_segmentation = None
                # if all_segmentation_data:
                #     batch_segmentation = [all_segmentation_data[j] for j in batch_indices]

                # combined_predictions = convert_prediction_to_dict(
                #     batch_prediction, 
                #     batch_paths, 
                #     output_debug_images=self.output_debug_images,
                #     segmentation_data=batch_segmentation,
                #     class_names=segmentation_class_names
                # )
                
                # # Create batch collection
                # self.result_queue.put({
                #     "type": "BATCH_READY",
                #     "data": combined_predictions,
                #     "batch_number": batch_number,
                #     "folder_name": folder_name,
                #     "batch_indices": batch_indices,
                #     "generate_mesh": self.generate_mesh,
                #     "filter_edges": self.filter_edges,
                #     "min_confidence": self.min_confidence
                # })
                self.handle_batch_result(batch_prediction, batch_indices, batch_number, folder_name, all_segmentation_data, segmentation_class_names)
            
            self.result_queue.put({"type": "DONE"})
            unload_if_overcommitted("post-metric-inference")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.result_queue.put({"type": "ERROR", "message": f"Failed to generate point cloud: {e}"})
        finally:
            self.stop_event.set()

    def modal(self, context, event):
        if event.type == 'TIMER':
            context.scene.da3_progress = self.progress
            context.scene.da3_progress_stage = self.stage
            
            if self.stage!= self.old_stage or self.progress != self.old_progress:
                self.old_stage = self.stage
                self.old_progress = self.progress
                # Force UI redraw
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
            
            # Check if thread died unexpectedly
            if self.thread and not self.thread.is_alive() and not self.stop_event.is_set():
                self.report({'ERROR'}, "Worker thread terminated unexpectedly")
                self.cleanup(context)
                return {'CANCELLED'}

            while not self.result_queue.empty():
                try:
                    msg = self.result_queue.get_nowait()
                except queue.Empty:
                    break
                
                if msg["type"] == "ERROR":
                    self.report({'ERROR'}, msg["message"])
                    self.cleanup(context)
                    return {'CANCELLED'}
                
                elif msg["type"] == "INIT_COLLECTION":
                    folder_name = msg["folder_name"]
                    self.parent_col = bpy.data.collections.new(folder_name)
                    context.scene.collection.children.link(self.parent_col)
                
                elif msg["type"] == "BATCH_READY":
                    if self.process_batch(context, msg) == {'CANCELLED'}:
                        self.cleanup(context)
                        return {'CANCELLED'}

                elif msg["type"] == "STREAMING_PLY":
                    if self._import_streaming_ply(context, msg) == {'CANCELLED'}:
                        self.cleanup(context)
                        return {'CANCELLED'}
                
                elif msg["type"] == "DONE":
                    self.report({'INFO'}, "Point cloud generation complete.")
                    self.cleanup(context)
                    return {'FINISHED'}
            
            if self.stop_event.is_set() and not self.thread.is_alive() and self.result_queue.empty():
                self.cleanup(context)
                return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.stop_event.set()

            self.cleanup(context)

            self.report({'WARNING'}, "Generation cancelled.")

            if self.thread.is_alive():
                self.thread.join()

            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def cleanup(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self.timer)
        # wm.progress_end()
        context.scene.da3_progress = -1

    def process_batch(self, context, msg):
        try:
            folder_name = msg["folder_name"]
            batch_number = msg["batch_number"]
            combined_predictions = msg["data"]
            batch_indices = msg["batch_indices"]
            generate_mesh = msg["generate_mesh"]
            filter_edges = msg["filter_edges"]
            min_confidence = msg["min_confidence"]
            
            parent_col = self.parent_col
            if not parent_col:
                parent_col = bpy.data.collections.new(folder_name)
                context.scene.collection.children.link(parent_col)
                
            batch_col_name = f"{folder_name}_Batch_{batch_number+1}"
            batch_col = bpy.data.collections.new(batch_col_name)
            parent_col.children.link(batch_col)
            
            if generate_mesh:
                import_mesh_from_depth(combined_predictions, collection=batch_col, filter_edges=filter_edges, min_confidence=min_confidence, global_indices=batch_indices)
            else:
                import_point_cloud(combined_predictions, collection=batch_col, filter_edges=filter_edges, min_confidence=min_confidence, global_indices=batch_indices)
            
            create_cameras(combined_predictions, collection=batch_col)
        except Exception as e:
            # end_progress_timer()
            import traceback
            print("DA3 ERROR while adding point clouds to Blender:")
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

    @classmethod
    def poll(cls, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name, context)
        return os.path.exists(model_path) and context.scene.da3_input_folder != ""