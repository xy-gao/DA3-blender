import os
import sys
from pathlib import Path

# Ensure local DA3 repo (and installed deps) are on sys.path before importing bundled modules
ADDON_ROOT = Path(__file__).parent
for p in (
    ADDON_ROOT / "deps_da3",
    ADDON_ROOT / "deps_public",
    ADDON_ROOT / "da3_repo",
    ADDON_ROOT / "da3_repo" / "da3_streaming",
    ADDON_ROOT / "da3_repo" / "da3_streaming" / "loop_utils",
):
    p_str = os.fspath(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)

# Dependencies.py already injects deps_public/deps_da3/DA3_DIR into sys.path on addon import.
# We import the streaming modules directly from the vendored repo folder.

# Map alternate checkpoint names to the config stem (mirrors operators.CONFIG_NAME_MAP)
CONFIG_NAME_MAP = {
    "da3-large-1.1": "da3-large",
    "da3-giant-1.1": "da3-giant",
    "da3nested-giant-large-1.1": "da3nested-giant-large",
}

def build_config(model_path: str, chunk_size: int, overlap: int, loop_chunk_size: int) -> dict:
    model_path = os.path.abspath(model_path)
    model_dir = Path(model_path).parent

    # Start from an in-memory default config (matching previous base_config_low_vram.yaml)
    cfg = {
        "Weights": {
            "DA3": model_path,
            "DA3_CONFIG": "",  # set below based on model stem
            "SALAD": "",        # set below if found/downloaded
        },
        "Model": {
            "chunk_size": max(1, int(chunk_size)),
            "overlap": max(1, int(overlap)),
            "loop_chunk_size": max(1, int(loop_chunk_size)),
            "loop_enable": True,
            "useDBoW": False,
            "delete_temp_files": True,
            "align_lib": "torch",
            "align_method": "sim3",
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": "saddle_balanced",
            "ref_view_strategy_loop": "saddle_balanced",
            "depth_threshold": 15.0,
            "save_depth_conf_result": True,
            "save_debug_info": False,
            "Sparse_Align": {
                "keypoint_select": "orb",
                "keypoint_num": 5000,
            },
            "IRLS": {
                "delta": 0.1,
                "max_iters": 5,
                "tol": "1e-9",
            },
            "Pointcloud_Save": {
                "sample_ratio": 1.0,
                "conf_threshold_coef": 0.75,
            },
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 32,
                "similarity_threshold": 0.85,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {
                "lang_version": "cpp",
                "max_iterations": 30,
                # Keep as string because Sim3LoopOptimizer expects to eval() this field
                "lambda_init": "1e-6",
            },
        },
    }

    # Choose config stem using the same mapping as operators.py
    model_stem = Path(model_path).stem
    config_stem = CONFIG_NAME_MAP.get(model_stem, model_stem)
    cfg_json = model_dir / f"{config_stem}.json"
    if not cfg_json.exists():
        # fall back to original stem
        cfg_json = model_dir / f"{model_stem}.json"
    cfg["Weights"]["DA3_CONFIG"] = os.fspath(cfg_json)

    # Point SALAD ckpt using the shared resolver; auto-download if missing (mirrors segmentation flow)
    try:
        from .operators import get_any_model_path, _URLS  # lazy import to avoid circular deps
        salad_path = Path(get_any_model_path("dino_salad.ckpt"))
        if not salad_path.exists():
            url = _URLS.get("dino_salad", "")
            if url:
                os.makedirs(salad_path.parent, exist_ok=True)
                print(f"Downloading dino_salad to {salad_path}...")
                import torch
                torch.hub.download_url_to_file(url, os.fspath(salad_path))
    except Exception as e:
        print(f"Warning: Failed to resolve/download dino_salad.ckpt via get_any_model_path: {e}")
        salad_path = Path(model_dir / "dino_salad.ckpt")

    if salad_path.exists():
        cfg["Weights"]["SALAD"] = os.fspath(salad_path)
    else:
        # If missing, disable loop to avoid crash; log a warning
        print("Warning: dino_salad.ckpt not found; disabling loop closure.")
        cfg["Model"]["loop_enable"] = False

    cfg["Model"]["chunk_size"] = max(1, int(chunk_size))
    cfg["Model"]["overlap"] = max(1, int(overlap))
    cfg["Model"]["loop_chunk_size"] = max(1, int(loop_chunk_size))
    cfg["Model"]["align_lib"] = "torch"
    return cfg

# From da3_streaming.py
def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


# From da3_streaming.py
def remove_duplicates(data_list):
    """
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result

# Based on DA3_Streaming from da3_streaming.py
class DA3_Modified_Streaming:
    def __init__(self, image_dir, save_dir, image_paths, config, model=None, progress_callback=None):
        self.config = config

        # Optional UI progress callback: callable(progress_float 0-100, message) -> bool to request stop
        self.progress_callback = progress_callback
        self._progress_done = 0.0
        self._progress_total = 1.0

        if not image_paths:
            raise ValueError("image_paths must be a non-empty list of image files")
        self.img_list = list(image_paths)

        if not save_dir:
            save_dir = os.path.join(image_dir, "debug_output")

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )

        self.img_dir = image_dir
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        # If a model instance is supplied, reuse it; otherwise load here.
        if model is not None:
            print("Using provided DepthAnything3 model instance.")
            self.model = model
        else:
            print("Loading model...")
            with open(self.config["Weights"]["DA3_CONFIG"]) as f:
                config = json.load(f)
            self.model = DepthAnything3(**config)
            weight = load_file(self.config["Weights"]["DA3"])
            self.model.load_state_dict(weight, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.skyseg_session = None

        self.chunk_indices = None  # [(begin_idx, end_idx), ...]

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
            self.loop_detector = LoopDetector(
                image_dir=image_dir, output=loop_info_save_path, config=self.config
            )
            self.loop_detector.load_model()

        print("init done.")

    def _emit_progress(self, inc=0.0, message=""):
        if self.progress_callback is None:
            return
        self._progress_done += inc
        frac = min(1.0, self._progress_done / max(1.0, self._progress_total))
        percent = frac * 100.0
        try:
            stop = self.progress_callback(percent, message)
        except Exception:
            return
        if stop:
            raise RuntimeError("Streaming cancelled by user")

    def get_loop_pairs(self):
        self.loop_detector.run()
        loop_list = self.loop_detector.get_loop_list()
        return loop_list

    def save_depth_conf_result(self, predictions, chunk_idx, s, R, T):
        if not self.config["Model"]["save_depth_conf_result"]:
            return
        os.makedirs(self.result_output_dir, exist_ok=True)

        chunk_start, chunk_end = self.chunk_indices[chunk_idx]

        if chunk_idx == 0:
            save_indices = list(range(0, chunk_end - chunk_start - self.overlap_e))
        elif chunk_idx == len(self.chunk_indices) - 1:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start))
        else:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start - self.overlap_e))

        print("[save_depth_conf_result] save_indices:")

        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            print(f"{global_idx}, ", end="")

            image = predictions.processed_images[local_idx]  # [H, W, 3] uint8
            depth = predictions.depth[local_idx]  # [H, W] float32
            conf = predictions.conf[local_idx]  # [H, W] float32
            intrinsics = predictions.intrinsics[local_idx]  # [3, 3] float32

            filename = f"frame_{global_idx}.npz"
            filepath = os.path.join(self.result_output_dir, filename)

            if self.config["Model"]["save_debug_info"]:
                np.savez_compressed(
                    filepath,
                    image=image,
                    depth=depth,
                    conf=conf,
                    intrinsics=intrinsics,
                    extrinsics=predictions.extrinsics[local_idx],
                    s=s,
                    R=R,
                    T=T,
                )
            else:
                np.savez_compressed(
                    filepath, image=image, depth=depth, conf=conf, intrinsics=intrinsics
                )
        print("")

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(chunk_image_paths)} images")

        ref_view_strategy = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
        ]

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = chunk_image_paths
                # images: ['xxx.png', 'xxx.png', ...]

                predictions = self.model.inference(images, ref_view_strategy=ref_view_strategy)

                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0

                print(predictions.processed_images.shape)  # [N, H, W, 3] uint8
                print(predictions.depth.shape)  # [N, H, W] float32
                print(predictions.conf.shape)  # [N, H, W] float32
                print(predictions.extrinsics.shape)  # [N, 3, 4] float32 (w2c)
                print(predictions.intrinsics.shape)  # [N, 3, 3] float32
        torch.cuda.empty_cache()

        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        np.save(save_path, predictions)

        return predictions

    def get_chunk_indices(self):
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                chunk_indices.append((start_idx, end_idx))
        return chunk_indices, num_chunks

    def align_2pcds(
        self,
        point_map1,
        conf1,
        point_map2,
        conf2,
        chunk1_depth,
        chunk2_depth,
        chunk1_depth_conf,
        chunk2_depth_conf,
    ):

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor_return, quality_score, method_used = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(
                f"[Depth Scale Precompute] scale: {scale_factor_return}, \
                    quality_score: {quality_score}, method_used: {method_used}"
            )
            scale_factor = scale_factor_return

        s, R, t = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:", t)

        return s, R, t

    def get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []
        for item in loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth, item[1].intrinsics, item[1].extrinsics
            )

            chunk_a_s = 0
            chunk_a_e = chunk_a_len = chunk_a_range[1] - chunk_a_range[0]
            chunk_b_s = -chunk_b_range[1] + chunk_b_range[0]
            chunk_b_e = point_map_loop_org.shape[0]
            chunk_b_len = chunk_b_range[1] - chunk_b_range[0]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

            print("chunk_a align")

            point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
            conf_loop = item[1].conf[chunk_a_s:chunk_a_e]
            print(self.chunk_indices[chunk_idx_a])
            print(chunk_a_range)
            print(chunk_a_rela_begin, chunk_a_rela_end)
            chunk_data_a = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                allow_pickle=True,
            ).item()

            point_map_a = depth_to_point_cloud_vectorized(
                chunk_data_a.depth, chunk_data_a.intrinsics, chunk_data_a.extrinsics
            )
            point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_a_depth = np.squeeze(chunk_data_a.depth[chunk_a_rela_begin:chunk_a_rela_end])
                chunk_a_depth_conf = np.squeeze(
                    chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_loop_depth = np.squeeze(item[1].depth[chunk_a_s:chunk_a_e])
                chunk_a_loop_depth_conf = np.squeeze(item[1].conf[chunk_a_s:chunk_a_e])
            else:
                chunk_a_depth = None
                chunk_a_loop_depth = None
                chunk_a_depth_conf = None
                chunk_a_loop_depth_conf = None

            s_a, R_a, t_a = self.align_2pcds(
                point_map_a,
                conf_a,
                point_map_loop_a,
                conf_loop,
                chunk_a_depth,
                chunk_a_loop_depth,
                chunk_a_depth_conf,
                chunk_a_loop_depth_conf,
            )

            print("chunk_b align")

            point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
            conf_loop = item[1].conf[chunk_b_s:chunk_b_e]
            print(self.chunk_indices[chunk_idx_b])
            print(chunk_b_range)
            print(chunk_b_rela_begin, chunk_b_rela_end)
            chunk_data_b = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                allow_pickle=True,
            ).item()

            point_map_b = depth_to_point_cloud_vectorized(
                chunk_data_b.depth, chunk_data_b.intrinsics, chunk_data_b.extrinsics
            )
            point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_b_depth = np.squeeze(chunk_data_b.depth[chunk_b_rela_begin:chunk_b_rela_end])
                chunk_b_depth_conf = np.squeeze(
                    chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_loop_depth = np.squeeze(item[1].depth[chunk_b_s:chunk_b_e])
                chunk_b_loop_depth_conf = np.squeeze(item[1].conf[chunk_b_s:chunk_b_e])
            else:
                chunk_b_depth = None
                chunk_b_loop_depth = None
                chunk_b_depth_conf = None
                chunk_b_loop_depth_conf = None

            s_b, R_b, t_b = self.align_2pcds(
                point_map_b,
                conf_b,
                point_map_loop_b,
                conf_loop,
                chunk_b_depth,
                chunk_b_loop_depth,
                chunk_b_depth_conf,
                chunk_b_loop_depth_conf,
            )

            print("a -> b SIM 3")
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", R_ab)
            print("Estimated Translation:", t_ab)

            loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        return loop_sim3_list

    def plot_loop_closure(
        self, input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
    ):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)

        # Visual in png format
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")
        for i, j, _ in self.loop_sim3_list:
            plt.plot(
                [x0[i], x0[j]],
                [y0[i], y0[j]],
                "r--",
                alpha=0.25,
                label="Loop (Before)" if i == 5 else "",
            )
            plt.plot(
                [x1[i], x1[j]],
                [y1[i], y1[j]],
                "g-",
                alpha=0.25,
                label="Loop (After)" if i == 5 else "",
            )
        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        # Plan progress steps: chunk inference + pairwise alignment + apply alignment; loop steps added later
        self._progress_total = num_chunks + max(0, num_chunks - 1) + num_chunks
        self._emit_progress(0, "Starting streaming")

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} \
                chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        pre_predictions = None
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")
            cur_predictions = self.process_single_chunk(
                self.chunk_indices[chunk_idx], chunk_idx=chunk_idx
            )
            torch.cuda.empty_cache()
            self._emit_progress(1, f"Processed chunk {chunk_idx}")

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(self.chunk_indices)-1})"
                )
                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                s, R, t = self.align_2pcds(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )
                self.sim3_list.append((s, R, t))
                self._emit_progress(1, f"Aligned chunks {chunk_idx-1}->{chunk_idx}")

            pre_predictions = cur_predictions

        if self.loop_enable:
            self.loop_list = self.get_loop_pairs()
            del self.loop_detector  # Save GPU Memory

            torch.cuda.empty_cache()

            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # Add loop-related progress steps (each loop chunk processed + final optimize)
            self._progress_total += len(loop_results) + 1
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True
                )

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
                self._emit_progress(1, "Processed loop pair")

            self.loop_sim3_list = self.get_loop_sim3_from_loop_predict(self.loop_predict_list)

            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            self._emit_progress(1, "Optimized loop closures")
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot

            self.plot_loop_closure(
                input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
            )

        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            aligned_chunk_data = {}

            aligned_chunk_data["world_points"] = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                aligned_chunk_data["world_points"], s, R, t
            )

            aligned_chunk_data["conf"] = chunk_data.conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, aligned_chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
                ).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
                points_first = depth_to_point_cloud_vectorized(
                    chunk_data_first.depth,
                    chunk_data_first.intrinsics,
                    chunk_data_first.extrinsics,
                )
                colors_first = chunk_data_first.processed_images
                confs_first = chunk_data_first.conf
                ply_path_first = os.path.join(self.pcd_dir, "0_pcd.ply")
                save_confident_pointcloud_batch(
                    points=points_first,  # shape: (H, W, 3)
                    colors=colors_first,  # shape: (H, W, 3)
                    confs=confs_first,  # shape: (H, W)
                    output_path=ply_path_first,
                    conf_threshold=np.mean(confs_first)
                    * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                    sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
                )
                if self.config["Model"]["save_depth_conf_result"]:
                    predictions = chunk_data_first
                    self.save_depth_conf_result(predictions, 0, 1, np.eye(3), np.array([0, 0, 0]))

            points = aligned_chunk_data["world_points"].reshape(-1, 3)
            colors = (aligned_chunk_data["images"].reshape(-1, 3)).astype(np.uint8)
            confs = aligned_chunk_data["conf"].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f"{chunk_idx+1}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                confs=confs,  # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

            if self.config["Model"]["save_depth_conf_result"]:
                predictions = chunk_data
                predictions.depth *= s
                self.save_depth_conf_result(predictions, chunk_idx + 1, s, R, t)

            self._emit_progress(1, f"Applied alignment chunk {chunk_idx+1}")

        self.save_camera_poses()

        # Ensure progress reaches 100%
        self._progress_done = self._progress_total
        self._emit_progress(0, "Done")

        print("Done.")

    def run(self):
        print(f"Using provided image list from {self.img_dir}...")
        print(f"Found {len(self.img_list)} images")
        self.process_long_sequence()

    def save_camera_poses(self):
        """
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        """
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]

        for i, idx in enumerate(
            range(first_chunk_range[0], first_chunk_range[1] - self.overlap_e)
        ):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1
            ]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            chunk_range_end = (
                chunk_range[1] - self.overlap_e
                if chunk_idx < len(self.all_camera_poses) - 1
                else chunk_range[1]
            )

            for i, idx in enumerate(range(chunk_range[0] + self.overlap_s, chunk_range_end)):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i + self.overlap_s]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i + self.overlap_s]

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        print(f"Camera poses saved to {poses_path}")

        intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
        with open(intrinsics_path, "w") as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f"{fx} {fy} {cx} {cy}\n")

        print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(
                    f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n"
                )

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        """
        Clean up temporary files and calculate reclaimed disk space.

        This method deletes all temporary files generated during processing from three directories:
        - Unaligned results
        - Aligned results
        - Loop results

        ~50 GiB for 4500-frame KITTI 00,
        ~35 GiB for 2700-frame KITTI 05,
        or ~5 GiB for 300-frame short seq.
        """
        if not self.delete_temp_files:
            return

        total_space = 0

        print(f"Deleting the temp files under {self.result_unaligned_dir}")
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_aligned_dir}")
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_loop_dir}")
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print("Deleting temp files done.")

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")

def run_streaming(
    image_dir: str,
    image_paths: list,
    output_dir: str | None,
    model_path: str,
    chunk_size: int,
    overlap: int,
    model=None,
    progress_callback=None,
) -> dict:
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not image_paths:
        raise ValueError("image_paths must be a non-empty list of image files")

    if not output_dir:
        output_dir = os.path.join(image_dir, "debug_output")
    os.makedirs(output_dir, exist_ok=True)

    loop_chunk_size = overlap
    config = build_config(model_path, chunk_size, overlap, loop_chunk_size)

    if config["Model"].get("align_lib", "") == "numba":
        warmup_numba()

    da3_streaming = DA3_Modified_Streaming(
        image_dir=image_dir,
        save_dir=output_dir,
        image_paths=image_paths,
        config=config,
        model=model,
        progress_callback=progress_callback,
    )
    da3_streaming.run()
    da3_streaming.close()

    pcd_dir = os.path.join(output_dir, "pcd")
    combined_ply = os.path.join(pcd_dir, "combined_pcd.ply")
    merge_ply_files(pcd_dir, combined_ply)

    return {"combined_ply": combined_ply, "pcd_dir": pcd_dir, "output_dir": output_dir}
