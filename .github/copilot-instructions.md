# DA3-blender – AI Coding Agent Instructions

These instructions are for AI coding agents working in this repo (Blender addon wrapping Depth-Anything-3).

## Big Picture
- **Goal:** Convert a folder of images (or video) into a complete, detailed, high-resolution 3D scene in Blender.
  - **Ideal Outcome:** Correctly scaled scene with separate, named objects. Static parts grouped and always visible; moving parts separated and animated. Output as textured 3D meshes (exportable) or point clouds.
  - **Current State:** Blender addon wrapping Depth Anything 3 (DA3) to import metric-aware point clouds/meshes + cameras. Includes streaming mode for improved alignment in image sequences.
- **Constraints & Challenges:**
  - **Hardware:** Must run on low VRAM (e.g., 4GB GTX 970). This necessitates strict batching (e.g., ~10 images at 504x280).
  - **Alignment:** DA3 aligns well within batches but poorly between batches. We currently separate geometry by batch to allow manual user tweaking. Streaming mode provides better inter-batch alignment for sequences.
  - **Artifacts:** Depth antialiasing at object edges creates "streamers" that require filtering.
  - **Model Capabilities:**
    - **Metric Model:** Provides metric depth & sky detection but **NO** camera intrinsics/extrinsics.
    - **Small/Base/Large/Giant:** Provide camera intrinsics/extrinsics & relative depth but **NO** sky detection or metric scale.
    - **Strategy:** We often need to combine outputs (e.g., use Large for cameras and Metric for scale).
- **Architecture:**
  - **Addon Layer:** `__init__.py`, `operators.py`, `panels.py`, `utils.py`, `dependencies.py`. Handles UI, threading, and Blender data creation.
  - **Library Layer:** `da3_repo` (vendored source) and `deps_da3`/`deps_public` (installed wheels). Includes `da3_streaming` for advanced sequence processing.
- **Runtime Flow:**
  1. `__init__.py` registers classes and checks dependencies via `dependencies.py`.
  2. `GeneratePointCloudOperator` (modal) starts a background thread for inference.
  3. Worker thread runs DA3 model (via `utils.run_model`) or streaming pipeline, handles batching/alignment, and puts results in a `queue`.
  4. Main thread (`modal` method) consumes queue to create Blender objects (Point Clouds, Meshes, Cameras).

## Key Patterns & Conventions

### 1. Threading & UI Responsiveness
- **Pattern:** Long-running tasks (Inference, Download) should preferably run in a separate thread to avoid freezing Blender's UI.
- **Implementation:**
  - `invoke()`: Initialize `threading.Thread`, `queue.Queue`, and `wm.event_timer_add`. Return `{'RUNNING_MODAL'}`.
  - `modal()`: On `TIMER` event, check `queue`. Process results (e.g., create objects) in the **main thread** (Blender API is not thread-safe).
  - **Example:** `GeneratePointCloudOperator` in `operators.py`.

### 2. Dependency Management (`dependencies.py`)
- **Mechanism:** Dependencies are installed locally into `deps_public` and `deps_da3` using `pip --target`.
- **Path Injection:** `sys.path` is modified at runtime to prioritize these local folders.
- **Rule:** Do not assume standard site-packages availability. Use `Dependencies.check()` and `Dependencies.install()`. Streaming dependencies require separate installation via `Dependencies.install_streaming_deps()`.

### 3. Coordinate Systems & Data Conversion
- **DA3/OpenCV:** Y-down, Z-forward.
- **Blender:** Z-up, Y-forward.
- **Conversion:**
  - Points: Swap Y/Z and invert Z. See `import_point_cloud` in `utils.py`.
  - Cameras: Apply -90 degree rotation on X axis. See `create_cameras` in `utils.py`.
- **Arrays:** Maintain `[N, H, W]` shape for depth/confidence maps.

### 4. Geometry Nodes & Materials
- **Point Clouds:** Created as Meshes with `GeometryNodes` modifier for rendering points.
- **Filtering:** `GeometryNodes` tree filters points dynamically based on `conf` attribute (Confidence).
- **Materials:** Procedural shader (`PointMaterial`) visualizes confidence or RGB colors.
- **Meshes:** `import_mesh_from_depth` creates textured meshes instead of point clouds.

## Key Files
- **`operators.py`**:
  - `GeneratePointCloudOperator`: Core logic. Handles batching (`skip_frames`, `overlap`), metric depth combination, segmentation/motion detection, and streaming mode selection.
  - `DownloadModelOperator`: Downloads weights to `models/`.
- **`utils.py`**:
  - `run_model`: Interface to DA3 API for standard inference.
  - `align_batches`: Aligns depth/extrinsics between batches using overlapping frames.
  - `import_point_cloud` / `import_mesh_from_depth`: Blender object creation.
  - `combine_base_and_metric`: Scales relative depth using metric model.
- **`streaming.py`**: Wrapper for DA3-Streaming pipeline, providing advanced sequence alignment and loop closure.
- **`__init__.py`**: Registry and `Scene` properties definitions. Keep minimal.

## Developer Workflows
- **Debugging:** Use `print()` which outputs to **Window > Toggle System Console**.
- **Model Management:** Models are cached in `operators.model`. Use `unload_current_model()` to free VRAM.
- **Streaming Mode:** For image sequences requiring better alignment, use streaming pipeline instead of batch processing. Requires separate dependency installation and config file generation.
- **New Features:**
  - Add UI props in `__init__.py`.
  - Implement logic in `operators.py` (threading) or `utils.py` (math/blender).
  - **Do not** modify `da3_repo` unless absolutely necessary.

## Specific Logic
- **Batching:** To support low VRAM, images are processed in batches. Alignment logic (`align_single_batch`) scales and transforms batches to match the previous one.
- **Metric Depth:** If enabled, runs a second pass with `da3metric-large` and scales the base model's output.
- **Segmentation/Motion:** Optional steps running YOLO or motion scoring before/after depth inference. Motion detection separates moving objects into animated point clouds.
- **Streaming:** Advanced mode using DA3-Streaming for video sequences. Provides loop closure and better inter-frame alignment. Configured via YAML files generated in `operators.py`.
