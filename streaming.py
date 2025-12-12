import os
from pathlib import Path
import yaml

from da3_repo.da3_streaming.da3_streaming import DA3_Streaming
from da3_repo.da3_streaming.loop_utils.config_utils import load_config
from da3_repo.da3_streaming.loop_utils.sim3utils import merge_ply_files, warmup_numba


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

    cfg["Weights"]["DA3"] = model_path
    model_stem = Path(model_path).stem
    cfg_json = os.path.join(os.path.dirname(model_path), f"{model_stem}.json")
    cfg["Weights"]["DA3_CONFIG"] = cfg_json

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
