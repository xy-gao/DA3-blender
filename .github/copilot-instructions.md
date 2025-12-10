# DA3-blender – AI Coding Agent Instructions

These instructions are for AI coding agents working in this repo (Blender addon wrapping Depth-Anything-3).

## Big Picture
- **Goal:** Blender addon to run Depth Anything 3 (DA3) on a folder of images and import a **metric-aware point cloud + cameras** as geometry nodes in Blender.
- **Two layers:**
  - **Addon layer (this repo root):** Blender UI, operators, dependency management, point-cloud import logic.
  - **DA3 library layer (`da3_repo` / `deps_da3`):** vendored Depth-Anything-3 repo used as a dependency; avoid modifying it unless explicitly asked.
- **Runtime flow:**
  1. `__init__.py` → `Dependencies.check()/install()` → sets up `deps_public`, `deps_da3`, `da3_repo` and installs Python deps.
  2. `register()` in `__init__.py` registers operators in `operators.py` and panel in `panels.py`, and adds `Scene` properties.
  3. User selects images folder + model in the DA3 panel and runs `GeneratePointCloudOperator`.
  4. `operators.get_model()` calls DA3 API (`depth_anything_3.api.DepthAnything3`), runs inference via `utils.run_model`, then `utils.convert_prediction_to_dict`, `utils.unproject_depth_map_to_point_map`, `utils.import_point_cloud`, and `utils.create_cameras` build Blender objects.

## Key Files & Responsibilities
- `__init__.py`
  - Defines `bl_info` and Blender `register()/unregister()` entry points.
  - Wires **scene-level properties**: `da3_input_folder`, `da3_model_name` enum, `da3_use_metric` bool.
  - **Pattern:** keep this file minimal and limited to registration + high-level dependency checks.
- `dependencies.py`
  - Handles **cloning** `Depth-Anything-3` into `da3_repo` and installing Python deps into `deps_public` and `deps_da3` using `pip --target`.
  - Injects `deps_public`, `deps_da3`, and `da3_repo` into `sys.path` before anything else imports DA3.
  - `Dependencies.check()` uses `pkg_resources.WorkingSet` and `requirements_for_check.txt` to validate minimal deps. Prefer updating `requirements_for_check.txt` when **required runtime packages** change.
  - Do **not** change clone URL or install strategy lightly; Blender users often have constrained Python environments.
- `operators.py`
  - Holds the two main operators:
    - `DownloadModelOperator`: downloads `.safetensors` weights into local `MODELS_DIR` using `torch.hub.download_url_to_file` using the `_URLS` map.
    - `GeneratePointCloudOperator`: runs DA3 inference on a folder and imports the result into Blender.
  - Global model cache (`model`, `current_model_name`) is used to **reuse a single DA3 model in VRAM** between runs; use `unload_current_model()` before loading another heavy model.
  - **Metric depth:** if `Scene.da3_use_metric` is true and `da3metric-large` weights are present, it runs a second model and calls `utils.combine_base_and_metric` to align and scale depth + extrinsics.
  - New operators should:
    - Inherit `bpy.types.Operator` and use `bl_idname` in the `da3.*` namespace.
    - Be **registered only** via `__init__.register()` to avoid double-registration.
- `panels.py`
  - Defines the **UI panel** `DA3Panel` in the 3D Viewport sidebar.
  - Uses `get_model_path()` from `operators.py` to show “ready” vs “download” state for both base and metric models.
  - **Nested metric logic:** hides `da3_use_metric` when `da3_model_name == "da3nested-giant-large"` since that model is intrinsically metric.
- `utils.py`
  - `run_model(target_dir, model)`: discovers images (`*.[jJpP][pPnN][gG]`) and calls `model.inference(image_paths)`.
  - `convert_prediction_to_dict(prediction, image_paths)`: standardizes DA3 prediction into a dict with numpy arrays and computes `world_points_from_depth` via `unproject_depth_map_to_point_map`.
  - `combine_base_and_metric(base, metric)`: DA3-specific alignment logic ported from `da3_repo` to scale base depth and extrinsics using metric model outputs and sky masks.
  - `import_point_cloud(d, collection)`: creates a `Points` mesh, sets up `FLOAT_COLOR` and `conf` attributes, and builds a **Geometry Nodes** tree that filters points by `conf` and applies proper material.
  - `create_cameras(predictions, collection, image_width, image_height)`: creates cameras from intrinsics/extrinsics, sets sensor/lens/shift, and names cameras using image filenames when available.
  - When updating anything here, preserve **array shape expectations** and coordinate system conversions; many pieces are interdependent.
- `da3_repo/` (Depth-Anything-3 upstream)
  - Treat as **third-party library**. Use its public API (`depth_anything_3.api.DepthAnything3`, configs) from the addon; do not change files here unless explicitly requested.

## Developer Workflows
- **Blender addon install & deps**
  - Users install the addon zip in Blender; on first enable, `Dependencies.install()` is triggered automatically and may take several minutes.
  - To debug dependency issues, log prints in `dependencies.py` are surfaced in Blender’s **System Console**.
- **Local Python / DA3 debugging (outside Blender)**
  - DA3 core lives in `da3_repo`. For non-Blender experiments, use its own `pyproject.toml`, `README.md`, and CLI docs under `docs/` and `src/`.
  - Keep **addon-facing utility code** in `utils.py` rather than duplicating DA3 internals here.

## Conventions & Patterns
- **Imports:**
  - Addon code imports DA3 as `from depth_anything_3.api import DepthAnything3` and utilities from `depth_anything_3.utils.alignment`.
  - Any new usage of DA3 should follow this pattern and **not** rely on private modules unless necessary.
- **Paths & models:**
  - Model weights live in a `models/` directory next to addon files; use `get_model_path(model_name)` and extend `_URLS` consistently for new models.
- **Blender data structures:**
  - Use `bpy.data.collections` and link objects to a per-run collection named after the input folder (see `GeneratePointCloudOperator`). New geometry/camera-generating features should reuse this pattern for organization.
  - Geometry Nodes graphs created in `import_point_cloud` should stay minimal and parameterized through **Group Inputs** (e.g., `Threshold`).
- **Error handling:**
  - Operators should use `self.report({"ERROR"|"WARNING"|"INFO"}, message)` and avoid raising bare exceptions; only throw in `Dependencies` / bootstrap code where Blender can’t continue.
  - For DA3 prediction issues, prefer printing a traceback (as in `GeneratePointCloudOperator.execute`) plus a concise user-facing error message.

## When Editing or Adding Code
- Prefer editing **addon wrapper code** (`__init__.py`, `operators.py`, `panels.py`, `utils.py`, `dependencies.py`) rather than the vendored DA3 core.
- When changing data flow between DA3 and Blender, verify these invariants:
  - `prediction.depth` → `[N, H, W]` float32 (torch or numpy), aligned with `prediction.extrinsics` and `prediction.intrinsics`.
  - `convert_prediction_to_dict` continues to populate `world_points_from_depth`, `images`, `depth`, `extrinsic`, `intrinsic`, `conf` with consistent shapes.
  - Cameras and point clouds remain in a **coherent coordinate system** (see the axis reordering and sign flip in `import_point_cloud`, and `T` + rotation in `create_cameras`).
- Before adding new features, consider whether they belong:
  - in `operators.py` (user-triggered actions),
  - in `utils.py` (math / conversion code), or
  - in DA3 core (`da3_repo`) as a reusable primitive.
