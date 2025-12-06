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
        
        # Model selection dropdown
        layout.prop(scene, "da3_model_name", text="Model")
        
        # Download button or status
        model_path = get_model_path(scene.da3_model_name)
        row = layout.row()
        if os.path.exists(model_path):
            row.label(text=f"Model {scene.da3_model_name} ready")
        else:
            row.operator("da3.download_model", text=f"Download {scene.da3_model_name}")

        # Metric model checkbox and download button/status
        if scene.da3_model_name != "da3nested-giant-large":
            layout.prop(scene, "da3_use_metric", text="Use Metric")
            if scene.da3_use_metric:
                # Metric combination mode
                layout.prop(scene, "da3_metric_mode", text="Metric Mode")

                metric_model_name = "da3metric-large"
                metric_model_path = get_model_path(metric_model_name)
                row = layout.row()
                if os.path.exists(metric_model_path):
                    row.label(text=f"Metric model {metric_model_name} ready")
                else:
                    op = row.operator("da3.download_model", text="Download Metric Model")
                    op.da3_override_model_name = metric_model_name

        layout.prop(scene, "da3_input_folder", text="Input Folder")
        layout.prop(scene, "da3_process_res", text="Process Resolution")
        layout.prop(scene, "da3_process_res_method", text="Resize Method")
        layout.prop(scene, "da3_batch_mode", text="Batch Mode")
        if scene.da3_batch_mode != "ignore_batch_size":
            layout.prop(scene, "da3_batch_size", text="Batch Size")
        layout.prop(scene, "da3_use_ray_pose", text="Use Ray-based Pose")
        layout.prop(scene, "da3_use_half_precision", text="Use Half Precision")
        layout.prop(scene, "da3_filter_edges", text="Filter Edges")
        layout.prop(scene, "da3_min_confidence", text="Min Confidence")
        layout.prop(scene, "da3_output_debug_images", text="Output Debug Images")
        row = layout.row()
        row.operator("da3.generate_point_cloud")
        row = layout.row()
        row.operator("da3.unload_model")