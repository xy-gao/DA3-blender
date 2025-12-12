import os
import sys
from pathlib import Path
import yaml

ADDON_ROOT = Path(__file__).parent

# Ensure local DA3 repo (and installed deps) are on sys.path so da3_streaming can be imported
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

# Dependencies.py already injects deps_public/deps_da3/DA3_DIR into sys.path on addon import.
# We import the streaming modules directly from the vendored repo folder.
import importlib

da3_streaming_mod = importlib.import_module("da3_streaming")
DA3_Streaming = da3_streaming_mod.DA3_Streaming
load_config = importlib.import_module("config_utils").load_config
merge_ply_files = importlib.import_module("sim3utils").merge_ply_files
warmup_numba = importlib.import_module("sim3utils").warmup_numba

# Map alternate checkpoint names to the config stem (mirrors operators.CONFIG_NAME_MAP)
CONFIG_NAME_MAP = {
    "da3-large-1.1": "da3-large",
    "da3-giant-1.1": "da3-giant",
    "da3nested-giant-large-1.1": "da3nested-giant-large",
}


def _load_base_config(base_cfg_path: Path) -> dict:
    if base_cfg_path.exists():
        return load_config(str(base_cfg_path))
    raise FileNotFoundError(f"Base streaming config not found at {base_cfg_path}")


def build_config(model_path: str, chunk_size: int, overlap: int, loop_chunk_size: int) -> dict:
    addon_root = Path(__file__).parent
    cfg_path = addon_root / "da3_repo" / "da3_streaming" / "configs" / "base_config_low_vram.yaml"
    if not cfg_path.exists():
        cfg_path = addon_root / "da3_repo" / "da3_streaming" / "configs" / "base_config.yaml"
    cfg = _load_base_config(cfg_path)

    model_path = os.path.abspath(model_path)
    model_dir = Path(model_path).parent

    cfg["Weights"]["DA3"] = model_path

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


def run_streaming(image_dir: str, output_dir: str, model_path: str, chunk_size: int, overlap: int) -> dict:
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")
    os.makedirs(output_dir, exist_ok=True)

    loop_chunk_size = overlap
    config = build_config(model_path, chunk_size, overlap, loop_chunk_size)

    if config["Model"].get("align_lib", "") == "numba":
        warmup_numba()

    da3_streaming = DA3_Streaming(image_dir, output_dir, config)
    da3_streaming.run()
    da3_streaming.close()

    pcd_dir = os.path.join(output_dir, "pcd")
    combined_ply = os.path.join(pcd_dir, "combined_pcd.ply")
    merge_ply_files(pcd_dir, combined_ply)

    return {"combined_ply": combined_ply, "pcd_dir": pcd_dir}
