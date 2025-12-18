import bpy
from .operators import get_model_path
import os

class DA3Panel(bpy.types.Panel):
    bl_label = "DA3"
    bl_idname = "VIEW3D_PT_da3"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DA3"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # --- Model (collapsible) ---
        model_box = layout.box()
        model_row = model_box.row()
        model_open = bool(getattr(scene, "da3_ui_model_open", True))
        model_row.prop(
            scene,
            "da3_ui_model_open",
            text="",
            emboss=False,
            icon='TRIA_DOWN' if model_open else 'TRIA_RIGHT',
        )
        model_row.label(text="Model")

        if model_open:
            # Model selection dropdown
            model_box.prop(scene, "da3_model_name", text="Model")

            # Download button or status
            model_path = get_model_path(scene.da3_model_name)
            row = model_box.row()
            if os.path.exists(model_path):
                row.label(text=f"Model {scene.da3_model_name} ready")
            else:
                row.operator("da3.download_model", text=f"Download {scene.da3_model_name}")

            # Precision toggle near model selection for clarity
            model_box.prop(scene, "da3_load_half_precision", text="Use FP16 Weights (experimental)")

            # Metric model checkbox and download button/status
            if scene.da3_model_name not in {"da3nested-giant-large", "da3nested-giant-large-1.1"}:
                model_box.prop(scene, "da3_use_metric", text="Use Metric")
                if scene.da3_use_metric:
                    model_box.prop(scene, "da3_metric_mode", text="Metric Mode")

                    metric_model_name = "da3metric-large"
                    metric_model_path = get_model_path(metric_model_name)
                    row = model_box.row()
                    if os.path.exists(metric_model_path):
                        row.label(text=f"Metric model {metric_model_name} ready")
                    else:
                        op = row.operator("da3.download_model", text="Download Metric Model")
                        op.da3_override_model_name = metric_model_name

        # Input folder stays prominent
        layout.prop(scene, "da3_input_folder", text="Input Folder")

        # --- Resolution (non-collapsible) ---
        res_box = layout.box()
        res_box.prop(scene, "da3_process_res", text="Process Resolution")
        res_box.prop(scene, "da3_process_res_method", text="Resize Method")

        # --- Batch (collapsible) ---
        batch_box = layout.box()
        batch_row = batch_box.row()
        batch_open = bool(getattr(scene, "da3_ui_batch_open", True))
        batch_row.prop(
            scene,
            "da3_ui_batch_open",
            text="",
            emboss=False,
            icon='TRIA_DOWN' if batch_open else 'TRIA_RIGHT',
        )
        batch_row.label(text="Batch")

        if batch_open:
            batch_box.prop(scene, "da3_batch_mode", text="Batch Mode")
            if scene.da3_batch_mode != "ignore_batch_size":
                batch_box.prop(scene, "da3_batch_size", text="Batch Size")
            if scene.da3_batch_mode != "skip_frames":
                batch_box.prop(scene, "da3_frame_stride", text="Frame Stride")
            batch_box.prop(scene, "da3_ref_view_strategy", text="Ref View Strategy")

            if scene.da3_batch_mode == "da3_streaming":
                box = batch_box.box()
                row = box.row()
                is_open = bool(getattr(scene, "da3_streaming_advanced_open", True))
                row.prop(
                    scene,
                    "da3_streaming_advanced_open",
                    text="",
                    emboss=False,
                    icon='TRIA_DOWN' if is_open else 'TRIA_RIGHT',
                )
                row.label(text="Advanced Streaming Options")

                if is_open:
                    box.prop(scene, "da3_streaming_output", text="Output")
                    box.prop(scene, "da3_streaming_loop_enable", text="Enable Loop Closure")
                    box.prop(scene, "da3_streaming_use_db_ow", text="Use DBOW")
                    box.prop(scene, "da3_streaming_align_lib", text="Alignment Library")
                    box.prop(scene, "da3_streaming_align_method", text="Alignment Method")
                    box.prop(scene, "da3_streaming_depth_threshold", text="Depth Threshold")
                    box.prop(scene, "da3_streaming_conf_threshold_coef", text="Conf Threshold Coef")
                    box.prop(scene, "da3_streaming_save_debug", text="Save Debug Info")
                    box.prop(scene, "da3_streaming_chunk_collections", text="Separate Chunk Collections")
        layout.prop(scene, "da3_use_ray_pose", text="Use Ray-based Pose")
        layout.prop(scene, "da3_filter_edges", text="Filter Edges")
        layout.prop(scene, "da3_min_confidence", text="Min Confidence")

        # --- Segmentation & Motion (collapsible) ---
        sm_box = layout.box()
        sm_row = sm_box.row()
        sm_open = bool(getattr(scene, "da3_ui_seg_motion_open", True))
        sm_row.prop(
            scene,
            "da3_ui_seg_motion_open",
            text="",
            emboss=False,
            icon='TRIA_DOWN' if sm_open else 'TRIA_RIGHT',
        )
        sm_row.label(text="Segmentation & Motion")

        if sm_open:
            sm_box.prop(scene, "da3_detect_motion", text="Detect Motion")
            if scene.da3_detect_motion:
                sm_box.prop(scene, "da3_motion_threshold", text="Motion Threshold")

            sm_box.prop(scene, "da3_use_segmentation")
            if scene.da3_use_segmentation:
                sm_box.prop(scene, "da3_segmentation_model")
                sm_box.prop(scene, "da3_segmentation_conf")
                sm_box.separator()

        layout.prop(scene, "da3_generate_mesh", text="Generate Meshes")
        layout.prop(scene, "da3_output_debug_images", text="Output Debug Images")
        row = layout.row()
        row.operator("da3.generate_point_cloud")
        row = layout.row()
        row.operator("da3.unload_model")

        # Progress bar
        if context.scene.da3_progress >= 0 and context.scene.da3_progress <= 100:
            layout.label(text=context.scene.da3_progress_stage)
            layout.progress(factor=context.scene.da3_progress / 100.0)