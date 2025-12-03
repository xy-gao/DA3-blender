import glob
import os
import numpy as np
import bpy
from mathutils import Matrix
import math
import torch

from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)

def unproject_depth_map_to_point_map(depth, extrinsics, intrinsics):
    N, H, W = depth.shape
    world_points = np.zeros((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels = np.stack([u, v, np.ones((H, W))], axis=-1).reshape(-1, 3)  # HW, 3
        invK = np.linalg.inv(intrinsics[i])
        rays = (invK @ pixels.T).T  # HW, 3
        depths = depth[i].reshape(-1)  # HW
        cam_points = rays * depths[:, np.newaxis]  # HW, 3
        cam_points_hom = np.hstack([cam_points, np.ones((len(depths), 1))])  # HW, 4
        E = np.vstack([extrinsics[i], [0, 0, 0, 1]])  # 4, 4
        cam_to_world = np.linalg.inv(E)
        world_points_hom = (cam_to_world @ cam_points_hom.T).T  # HW, 4
        world_points_i = world_points_hom[:, :3] / world_points_hom[:, 3:4]
        world_points[i] = world_points_i.reshape(H, W, 3)
    return world_points

def run_model(image_paths, model, process_res=504, process_res_method="upper_bound_resize", use_half=False, use_ray_pose=False):
    print(f"Processing {len(image_paths)} images")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1024**2
        free, total = torch.cuda.mem_get_info()
        free_mb = free / 1024**2
        total_mb = total / 1024**2
        print(f"VRAM before inference: {allocated:.1f} MB (free: {free_mb:.1f} MB / {total_mb:.1f} MB)")
    import torch.cuda.amp as amp
    if use_half:
        with amp.autocast():
            prediction = model.inference(image_paths, process_res=process_res, process_res_method=process_res_method, use_ray_pose=use_ray_pose)
    else:
        prediction = model.inference(image_paths, process_res=process_res, process_res_method=process_res_method, use_ray_pose=use_ray_pose)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        allocated = torch.cuda.memory_allocated() / 1024**2
        free, total = torch.cuda.mem_get_info()
        free_mb = free / 1024**2
        total_mb = total / 1024**2
        print(f"VRAM after inference: {allocated:.1f} MB (peak: {peak:.1f} MB, free: {free_mb:.1f} MB / {total_mb:.1f} MB)")
    # DEBUG: inspect prediction object for this model
    print("DEBUG prediction type:", type(prediction))
    if hasattr(prediction, "__dict__"):
        print("DEBUG prediction.__dict__ keys:", list(prediction.__dict__.keys()))
    else:
        print("DEBUG dir(prediction):", dir(prediction))
    return prediction

# Helper functions for matrix operations and type conversion
def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.array(x)

def _extrinsic_to_4x4_torch(ext_3x4):
    if ext_3x4.shape == (3, 4):
        last_row = torch.tensor([0, 0, 0, 1], device=ext_3x4.device, dtype=ext_3x4.dtype)
        return torch.cat([ext_3x4, last_row.unsqueeze(0)], dim=0)
    return ext_3x4

def _invert_4x4_torch(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def _extrinsic_to_4x4_numpy(ext_3x4):
    R = ext_3x4[:, :3]
    t = ext_3x4[:, 3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def _invert_4x4_numpy(T_4x4):
    R = T_4x4[:3, :3]
    t = T_4x4[:3, 3]
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

# Transform and scale each batch to align with previous batch
# all_predictions is list of (prediction_for_batch, frame_indices_for_batch)
# prediction_for_batch is the result returned by run_model, and is class depth_anything_3.specs.Prediction
# and has these fields: ['depth', 'is_metric', 'sky', 'conf', 'extrinsics', 'intrinsics', 'processed_images', 'gaussians', 'aux', 'scale_factor']
def align_batches(all_predictions):
    if not all_predictions:
        return []

    # result, a list of predictions with aligned extrinsics and depths
    aligned_predictions = []
    
    # First batch doesn't need aligning
    first_pred, first_indices = all_predictions[0]
    aligned_predictions.append(first_pred)
    prev_pred = first_pred
    prev_indices = first_indices
    
    # Loop through the rest of the batches
    for i in range(1, len(all_predictions)):
        curr_pred_orig, curr_indices = all_predictions[i]
        
        # Shallow copy to avoid modifying original
        import copy
        curr_pred = copy.copy(curr_pred_orig)
        
        curr_depth = _to_tensor(curr_pred.depth).float() # depth of every pixel in every image in the batch
        curr_conf = _to_tensor(curr_pred.conf).float() # confidence of depth of every pixel in every image in the batch, range 0 to more than 1
        curr_ext = _to_tensor(curr_pred.extrinsics) # camera position and rotation for every image in the batch (or None for Metric/Mono model)
        if curr_ext is not None:
            curr_ext = curr_ext.float()
        
        # Alignment for Metric/Mono model is not supported yet. TODO: still align the depth based on overlap images
        if curr_ext is None:
            print(f"Batch {i} has no extrinsics, skipping alignment.")
            aligned_predictions.append(curr_pred)
            prev_pred = curr_pred
            prev_indices = curr_indices
            continue

        # depths, depth confidences, and camera poses for all images in the previous batch
        prev_depth = _to_tensor(prev_pred.depth).float()
        prev_conf = _to_tensor(prev_pred.conf).float()
        prev_ext = _to_tensor(prev_pred.extrinsics).float()
        
        # Find overlapping indices
        common_indices = set(prev_indices) & set(curr_indices)
        if not common_indices:
            print(f"Warning: Batch {i} has no overlap with Batch {i-1}. Alignment may be poor.")
            aligned_predictions.append(curr_pred)
            prev_pred = curr_pred
            prev_indices = curr_indices
            continue
            
        # Sort common indices to ensure deterministic order
        common_indices = sorted(list(common_indices))
        
        # Collect valid pixels for depth scaling
        valid_prev_depths = []
        valid_curr_depths = []
        
        # Collect transforms for extrinsic alignment
        transforms = []

        # for each overlapping frame
        for global_idx in common_indices:
            # Find local index in prev and curr
            idx_prev = prev_indices.index(global_idx)
            idx_curr = curr_indices.index(global_idx)
            
            d_prev = prev_depth[idx_prev] # [H, W] depth of every pixel for this frame in the previous batch
            d_curr = curr_depth[idx_curr] # [H, W] depth of every pixel for this frame in the current batch
            c_prev = prev_conf[idx_prev]  # [H, W] confidence of every pixel for this frame in the previous batch
            
            # We only want to calculate scale from pixels that aren't sky
            # For Metric/Mono/Nested models use the returned sky mask
            # For base models there is no sky mask, so assume all pixels are non-sky
            non_sky_mask = torch.ones_like(d_prev, dtype=torch.bool) # [H, W]
            if hasattr(prev_pred, 'sky') and prev_pred.sky is not None:
                 non_sky_mask = non_sky_mask & compute_sky_mask(_to_tensor(prev_pred.sky)[idx_prev], threshold=0.3)
            if hasattr(curr_pred, 'sky') and curr_pred.sky is not None:
                 non_sky_mask = non_sky_mask & compute_sky_mask(_to_tensor(curr_pred.sky)[idx_curr], threshold=0.3)
            
            # Use compute_alignment_mask for robust pixel selection
            # Ensure inputs are at least 3D [1, H, W] for the utils
            d_prev_3d = d_prev.unsqueeze(0)
            d_curr_3d = d_curr.unsqueeze(0)
            c_prev_3d = c_prev.unsqueeze(0)
            non_sky_mask_3d = non_sky_mask.unsqueeze(0)
            
            c_prev_ns = c_prev[non_sky_mask] # [num_non_sky_pixels]
            if c_prev_ns.numel() > 0:
                c_prev_sampled = sample_tensor_for_quantile(c_prev_ns, max_samples=100000) # if there are more than 100,000 non-sky pixels, randomly select 100,000 of them
                median_conf = torch.quantile(c_prev_sampled, 0.5) # calculate the median confidence (half the pixels have higher confidence than this, half have lower confidence)

                # DA3 function, mask array is true for pixels that aren't sky and whose confidence is better than half the other non-sky pixels
                mask_3d = compute_alignment_mask(
                    c_prev_3d, non_sky_mask_3d, d_prev_3d, d_curr_3d, median_conf
                ) # [1, H, W] boolean mask
                mask = mask_3d.squeeze(0) # [H, W]
            else:
                mask = non_sky_mask # [H, W]

            # make sure there are at least 11 valid pixels (ie. there were originally at least 22 non-sky pixels before we chose the best half)
            if mask.sum() > 10:
                valid_prev_depths.append(d_prev[mask]) # [num_valid_pixels]
                valid_curr_depths.append(d_curr[mask]) # [num_valid_pixels]
            
            E_prev = _extrinsic_to_4x4_torch(prev_ext[idx_prev]) # 4x4 camera transform matrix for this frame in previous batch
            E_curr = _extrinsic_to_4x4_torch(curr_ext[idx_curr]) # 4x4 camera transform matrix for this frame in current batch
            
            transforms.append((E_prev, E_curr))

        # All overlap frames have now been processed
        # Compute global scale factor
        if valid_prev_depths:
            all_prev = torch.cat(valid_prev_depths) # [total_valid_pixels]
            all_curr = torch.cat(valid_curr_depths) # [total_valid_pixels]
            scale = least_squares_scale_scalar(all_curr, all_prev) # [1], DA3 function, current batch depth / previous batch depth, with the least error squared
        else:
            scale = torch.tensor(1.0) # 1x scale if there were no overlap frames with at least 22 non-sky pixels
            
        scale_val = float(scale.item())
        print(f"Batch {i} alignment: scale={scale_val}")
        
        # Apply scale to current depth
        curr_pred.depth = _to_numpy(curr_depth * scale)
        
        # Compute similarity transform T that maps P_world_curr to P_world_prev
        # P_world_prev = T @ P_world_curr
        # We derive T from: E_prev @ P_world_prev = s * E_curr @ P_world_curr
        # E_prev @ T @ P_world_curr = S @ E_curr @ P_world_curr
        # T = inv(E_prev) @ S @ E_curr
        
        # 4x4 scaling matrix (based on all overlap frames combined)
        S = torch.eye(4, device=curr_depth.device, dtype=curr_depth.dtype)
        S[0, 0] = scale
        S[1, 1] = scale
        S[2, 2] = scale
        
        # List of 4x4 rigid transform matrices (one per valid overlap frame)
        # T is the rigid alignment that maps curr world coords to prev world coords
        candidate_Ts = []
        
        # Find rigid transform for each overlap frame: T = inv(E_prev) @ E_curr
        # This transforms points from curr batch's world frame to prev batch's world frame
        for E_prev, E_curr in transforms:
            # T = inv(E_prev) @ E_curr (rigid alignment, no scaling here)
            T = _invert_4x4_torch(E_prev) @ E_curr
            candidate_Ts.append(T)
            
        # Select best T (minimize error across all overlapping frames)
        best_T = None
        min_error = float('inf')
        
        if len(candidate_Ts) == 1:
            best_T = candidate_Ts[0]
        elif len(candidate_Ts) > 1:
            for idx, T_cand in enumerate(candidate_Ts):
                error = 0.0
                T_cand_inv = _invert_4x4_torch(T_cand)
                
                for E_prev, E_curr in transforms:
                    # Check consistency: after applying T, E_curr should align with E_prev
                    # E_aligned = E_curr @ inv(T)
                    # We compare E_aligned with E_prev (before scaling)
                    
                    E_aligned = E_curr @ T_cand_inv
                    
                    # Error = distance between aligned translation and E_prev translation
                    dist = torch.norm(E_aligned[:3, 3] - E_prev[:3, 3]).item()
                    error += dist
                
                print(f"  Candidate {idx} error: {error:.4f}")
                if error < min_error:
                    min_error = error
                    best_T = T_cand
            
            print(f"Batch {i} alignment: Best T error = {min_error:.4f} (selected from {len(candidate_Ts)} candidates)")
        else:
            best_T = torch.eye(4, device=curr_depth.device, dtype=curr_depth.dtype)
            # If no overlap, we still need to apply scale to extrinsics?
            # If we just scale depth, we should scale extrinsics translation too.
            # T = S (pure scaling)
            best_T = S

        T_align = best_T
        T_align_inv = _invert_4x4_torch(T_align)

        # Apply alignment and scaling to all extrinsics in curr_batch
        # 1. First align: E_aligned = E_curr @ inv(T) - transforms to prev batch's world frame
        # 2. Then scale translation: the camera positions need to be scaled by S
        #    to match the scaled depths
        
        new_extrinsics = []
        for ext_3x4 in curr_ext:
            E_curr = _extrinsic_to_4x4_torch(ext_3x4)
            
            # Align to previous batch's world frame
            E_aligned = E_curr @ T_align_inv
            
            # Scale the translation (camera position) to match scaled depths
            # E is world-to-camera, so we need to scale the translation appropriately
            # For world-to-camera: t_new = s * t (scale camera position in world)
            E_new = E_aligned.clone()
            E_new[:3, 3] = E_aligned[:3, 3] * scale
            
            new_extrinsics.append(E_new[:3, :4])
            
        curr_pred.extrinsics = _to_numpy(torch.stack(new_extrinsics))
        
        # Add the aligned prediction for this batch to the result list
        aligned_predictions.append(curr_pred)
        prev_pred = curr_pred
        prev_indices = curr_indices
        
    # We've finished all batches, return a list of aligned predictions
    return aligned_predictions

def combine_overlapping_predictions(all_predictions, full_image_paths):
    if not all_predictions:
        raise ValueError("No predictions to combine")

    # all_predictions: list of (prediction, index_list) where index_list holds
    # the global frame indices used for that batch. First batch is contiguous;
    # later batches include overlap frames plus new frames.
    all_images = []
    all_depths = []
    all_extrinsics = []
    all_intrinsics = []
    all_confs = []

    prev_extrinsics = None
    prev_overlap_idx = None

    for i, (prediction, index_list) in enumerate(all_predictions):
        # Map batch indices back to image_paths just for logging/consistency
        batch_paths = [full_image_paths[j] for j in index_list]
        pred_dict = convert_prediction_to_dict(prediction, batch_paths)
        
        images_raw = np.asarray(prediction.processed_images).astype(np.float32)  # Raw 0-255, ensure numpy
        depths = pred_dict['depth']
        extrinsics = pred_dict['extrinsic']  # List of 3x4
        intrinsics = pred_dict['intrinsic']
        confs = pred_dict['conf']
        
        if i == 0:
            # First batch, use as is
            all_images.append(images_raw)
            all_depths.append(depths)
            all_extrinsics.extend(extrinsics)
            all_intrinsics.extend(intrinsics)
            all_confs.append(confs)
            prev_extrinsics = extrinsics
            prev_overlap_idx = len(extrinsics) - 1  # Last frame of this batch
        else:
            # Subsequent batches, adjust extrinsics
            # Overlap frame is the first in this batch
            overlap_current_3x4 = extrinsics[0]
            overlap_prev_3x4 = prev_extrinsics[prev_overlap_idx]
            
            # Work in camera-to-world space for alignment
            Ec_curr = _extrinsic_to_4x4_numpy(overlap_current_3x4)   # world->cam (current overlap)
            Ec_prev = _extrinsic_to_4x4_numpy(overlap_prev_3x4)      # world->cam (prev overlap)
            Cc_curr = _invert_4x4_numpy(Ec_curr)  # cam->world
            Cc_prev = _invert_4x4_numpy(Ec_prev)  # cam->world

            # Transform that maps current cameras into previous world frame
            # C_prev ≈ T @ C_curr  =>  T = C_prev @ inv(C_curr)
            T_align = Cc_prev @ _invert_4x4_numpy(Cc_curr)

            # Compare average depth for overlapping frames across batches
            base_depth = depths[0]
            prev_batch_depth = all_depths[-1][-1]  # last frame depth of previous batch
            prev_mean = float(prev_batch_depth.mean())
            curr_mean = float(base_depth.mean())
            print("DEBUG overlap depth means:")
            print("  prev batch overlap mean depth:", prev_mean)
            print("  current batch overlap mean depth:", curr_mean)

            # Compute a simple scale factor so the mean depth of the
            # current batch's overlap frame matches the previous batch.
            scale = prev_mean / curr_mean if curr_mean > 1e-6 else 1.0
            print("  applied depth scale:", scale)

            # Scale all depths in this batch so that geometry matches across batches
            depths = depths * scale

            # Apply transform in cam->world space, then convert back to world->cam extrinsics.
            # NOTE: camera translation encodes depth scale, so we also scale translations
            # when aligning, to stay consistent with depth scaling.
            adjusted_extrinsics = []
            for ext_3x4 in extrinsics:
                Ec = _extrinsic_to_4x4_numpy(ext_3x4)   # world->cam
                Cc = _invert_4x4_numpy(Ec)              # cam->world
                # Scale camera position before applying global alignment
                Cc_scaled = Cc.copy()
                Cc_scaled[:3, 3] *= scale
                Cc_aligned = T_align @ Cc_scaled
                Ec_aligned = _invert_4x4_numpy(Cc_aligned)
                adjusted_3x4 = np.hstack([Ec_aligned[:3, :3], Ec_aligned[:3, 3:4]])
                adjusted_extrinsics.append(adjusted_3x4)
            
            # Skip the first (overlap) for all
            all_images.append(images_raw[1:])
            all_depths.append(depths[1:])
            all_extrinsics.extend(adjusted_extrinsics[1:])
            all_intrinsics.extend(intrinsics[1:])
            all_confs.append(confs[1:])
            
            # Update prev
            prev_extrinsics = adjusted_extrinsics
            prev_overlap_idx = len(adjusted_extrinsics) - 1
    
    # Concatenate
    combined = type('CombinedPrediction', (), {})()
    combined.processed_images = np.concatenate(all_images, axis=0)
    combined.depth = np.concatenate(all_depths, axis=0)
    combined.extrinsics = all_extrinsics  # Note: plural, as in original
    combined.intrinsics = all_intrinsics
    combined.conf = np.concatenate(all_confs, axis=0)
    
    # Tint batches for debugging alignment (keep brightness, change hue per batch)
    tints = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # red, green, blue, yellow
    batch_effective_frames = [len(all_predictions[0][0].extrinsics)]
    for i in range(1, len(all_predictions)):
        # Subsequent batches contribute all but the first (overlap) frame
        batch_effective_frames.append(len(all_predictions[i][0].extrinsics) - 1)

    start_frame = 0
    for batch_idx, frames in enumerate(batch_effective_frames):
        end_frame = start_frame + frames
        batch_images = combined.processed_images[start_frame:end_frame]  # [frames, H, W, 3]
        if frames > 0:
            luminance = batch_images.mean(axis=-1, keepdims=True) / 255.0  # 0-1
            tint = np.array(tints[batch_idx % len(tints)], dtype=np.float32)
            tinted = luminance * tint
            combined.processed_images[start_frame:end_frame] = tinted * 255.0
        start_frame = end_frame
    
    return combined

def convert_prediction_to_dict(prediction, image_paths=None):
    predictions = {}

    # images is already numpy in your current pipeline
    predictions['images'] = prediction.processed_images.astype(np.float32) / 255.0  # [N, H, W, 3]

    # depth / extrinsics / intrinsics may be torch tensors after combination; ensure numpy
    predictions['depth'] = _to_numpy(prediction.depth)
    predictions['extrinsic'] = _to_numpy(prediction.extrinsics)
    predictions['intrinsic'] = _to_numpy(prediction.intrinsics)
    predictions['conf'] = prediction.conf

    if image_paths is not None:
        predictions['image_paths'] = image_paths
    
    print("DEBUG shapes:")
    print("  images:", predictions['images'].shape)
    print("  depth:", predictions['depth'].shape)
    print("  extrinsic:", np.array(predictions['extrinsic']).shape)
    print("  intrinsic:", np.array(predictions['intrinsic']).shape)
    print("Computing world points from depth map...")

    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise ValueError("Prediction has no camera parameters; cannot create world-space point cloud.")

    world_points = unproject_depth_map_to_point_map(
        predictions['depth'],
        predictions['extrinsic'],
        predictions['intrinsic'],
    )
    predictions["world_points_from_depth"] = world_points
    return predictions

# Based on da3_repo/src/depth_anything_3/model/da3.py, simplified for this addon
def combine_base_and_metric(base, metric):
    """
    Combine a base DA3 prediction (with camera poses) and a metric DA3 prediction (no poses)
    to produce a metric-scaled depth and adjusted extrinsics.

    Assumes shapes:
      - base.depth:        [B, H, W]
      - metric.depth:      [B, H, W]
      - base.intrinsics:   [B, 3, 3]
      - base.extrinsics:   [N, 3, 4] or [B, N, 4, 4]
      - metric.sky:        [B, H, W]
    depth_conf is optional and ignored for scaling.
    """
    output = base  # work in-place on base prediction

    # Pull relevant fields
    depth = output.depth          # [B, H, W] (torch or numpy)
    metric_depth = metric.depth   # [B, H, W]
    intrinsics = output.intrinsics  # [B, 3, 3]
    sky = metric.sky              # [B, H, W]

    depth = _to_tensor(depth)
    metric_depth = _to_tensor(metric_depth)
    intrinsics = _to_tensor(intrinsics)
    sky = _to_tensor(sky)

    # Ensure everything is float
    depth = depth.float()
    metric_depth = metric_depth.float()
    intrinsics = intrinsics.float()
    sky = sky.float()

    # Add channel dim so depth/metric_depth become [B, 1, H, W]
    if depth.ndim == 3:
        depth_4d = depth.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    if metric_depth.ndim == 3:
        metric_4d = metric_depth.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected metric_depth shape: {metric_depth.shape}")

    # Metric scaling: intrinsics must be (B, N, 3, 3)
    # Here N=1 (single view per batch entry), so we add that dim.
    ixt = intrinsics  # [B, 3, 3]
    if ixt.ndim == 3:
        ixt_4d = ixt.unsqueeze(1)  # [B, 1, 3, 3]
    else:
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")

    metric_4d = apply_metric_scaling(metric_4d, ixt_4d)  # [B, 1, H, W] (scaled metric depth)

    # Back to [B, H, W]
    depth = depth_4d.squeeze(1)
    metric_depth = metric_4d.squeeze(1)

    # Non-sky mask: [B, H, W]
    non_sky_mask = compute_sky_mask(sky, threshold=0.3)
    assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"

    # Align using compute_alignment_mask logic from da3.py
    # output.conf is the depth confidence
    
    # Sample depth confidence for quantile computation
    depth_conf_ns = output.conf[non_sky_mask]
    depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
    median_conf = torch.quantile(depth_conf_sampled, 0.5)

    # Compute alignment mask
    align_mask = compute_alignment_mask(
        output.conf, non_sky_mask, depth, metric_depth, median_conf
    )

    # Compute scale factor using least squares on aligned pixels
    valid_depth = depth[align_mask]
    valid_metric_depth = metric_depth[align_mask]

    scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

    # Apply scale to depth
    depth = depth * scale_factor

    # Scale extrinsics translation
    extrinsics = _to_tensor(output.extrinsics)
    print("DEBUG combine_base_and_metric: extrinsics shape:", extrinsics.shape)

    if extrinsics.ndim == 3:
        # [N, 3, 4]: scale translation column
        extrinsics = extrinsics.float()
        extrinsics[:, :, 3] = extrinsics[:, :, 3] * scale_factor
    elif extrinsics.ndim == 4:
        # [B, N, 4, 4]
        extrinsics = extrinsics.float()
        extrinsics[:, :, :3, 3] = extrinsics[:, :, :3, 3] * scale_factor
    else:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    # Optional: handle sky regions roughly like nested model (set sky to far depth)
    non_sky_depth = depth[non_sky_mask]
    if non_sky_depth.numel() > 100000:
        idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
        sampled_depth = non_sky_depth[idx]
    else:
        sampled_depth = non_sky_depth

    non_sky_max = torch.quantile(sampled_depth, 0.99)
    non_sky_max = torch.minimum(non_sky_max, torch.tensor(200.0, device=depth.device))

    # We don't have depth_conf; use a dummy one and ignore it after
    depth_4d = depth.unsqueeze(1)
    dummy_conf = torch.ones_like(depth_4d)
    depth_4d, _ = set_sky_regions_to_max_depth(
        depth_4d, dummy_conf, non_sky_mask.unsqueeze(1), max_depth=non_sky_max
    )
    depth = depth_4d.squeeze(1)

    # Write back into output prediction
    output.depth = depth
    output.extrinsics = extrinsics
    output.is_metric = 1
    # least_squares_scale_scalar returns a scalar tensor
    output.scale_factor = float(scale_factor.item())

    return output


def combine_base_with_metric_depth(base, metric):
    """Combine base prediction cameras with raw metric model depth.

    This variant keeps **base intrinsics/extrinsics/conf** but **replaces
    depth with metric.depth in metres**, then applies the same sky-handling
    logic as `combine_base_and_metric`.

    Assumes shapes:
      - base.depth:        [B, H, W]
      - metric.depth:      [B, H, W]
      - base.intrinsics:   [B, 3, 3]
      - base.extrinsics:   [N, 3, 4] or [B, N, 4, 4]
      - metric.sky:        [B, H, W]
    """
    output = base

    # Base / metric depths and sky mask
    base_depth = _to_tensor(base.depth).float()        # [B, H, W]
    metric_depth = _to_tensor(metric.depth).float()      # [B, H, W]
    sky = _to_tensor(metric.sky).float()                 # [B, H, W]

    if base_depth.ndim != 3 or metric_depth.ndim != 3:
        raise ValueError(
            f"Unexpected depth shapes: base={base_depth.shape}, metric={metric_depth.shape}"
        )

    # Non-sky mask and basic sanity check
    non_sky_mask = compute_sky_mask(sky, threshold=0.3)
    if non_sky_mask.sum() <= 10:
        raise ValueError("Insufficient non-sky pixels for metric depth sky handling")

    # Compute global scale factor aligning base depth to metric depth
    # Use robust alignment mask
    depth_conf_ns = output.conf[non_sky_mask]
    depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
    median_conf = torch.quantile(depth_conf_sampled, 0.5)

    align_mask = compute_alignment_mask(
        output.conf, non_sky_mask, base_depth, metric_depth, median_conf
    )

    valid_base = base_depth[align_mask]
    valid_metric = metric_depth[align_mask]
    scale_factor = least_squares_scale_scalar(valid_metric, valid_base)

    # Use metric depth as final depth (in metres)
    depth = metric_depth

    # Estimate a far depth for sky regions
    non_sky_depth = depth[non_sky_mask]
    if non_sky_depth.numel() > 100000:
        idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
        sampled_depth = non_sky_depth[idx]
    else:
        sampled_depth = non_sky_depth

    non_sky_max = torch.quantile(sampled_depth, 0.99)
    non_sky_max = torch.minimum(non_sky_max, torch.tensor(200.0, device=depth.device))

    depth_4d = depth.unsqueeze(1)
    dummy_conf = torch.ones_like(depth_4d)
    depth_4d, _ = set_sky_regions_to_max_depth(
        depth_4d, dummy_conf, non_sky_mask.unsqueeze(1), max_depth=non_sky_max
    )
    depth = depth_4d.squeeze(1)

    # Scale base extrinsics translation so cameras match metric scale
    extrinsics = _to_tensor(output.extrinsics)
    print("DEBUG combine_base_with_metric_depth: extrinsics shape:", extrinsics.shape)

    if extrinsics.ndim == 3:
        extrinsics = extrinsics.float()
        extrinsics[:, :, 3] = extrinsics[:, :, 3] * scale_factor
    elif extrinsics.ndim == 4:
        extrinsics = extrinsics.float()
        extrinsics[:, :, :3, 3] = extrinsics[:, :, :3, 3] * scale_factor
    else:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    # Write back into output: metric depth + scaled base cameras
    output.depth = depth
    output.extrinsics = extrinsics
    output.is_metric = 1
    output.scale_factor = float(scale_factor.item())

    return output

def import_point_cloud(d, collection=None):
    points = d["world_points_from_depth"]
    images = d["images"]
    conf = d["conf"]
    points_batch = points.reshape(-1, 3)
    reordered_points_batch = points_batch.copy()
    reordered_points_batch[:, [0, 1, 2]] = points_batch[:, [0, 2, 1]]
    reordered_points_batch[:, 2] = -reordered_points_batch[:, 2]
    points_batch = reordered_points_batch
    colors_batch = images.reshape(-1, 3)
    colors_batch = np.hstack((colors_batch, np.ones((colors_batch.shape[0], 1))))
    conf_batch = conf.reshape(-1)
    mesh = bpy.data.meshes.new(name="Points")
    vertices = points_batch.tolist()
    mesh.from_pydata(vertices, [], [])
    attribute = mesh.attributes.new(name="point_color", type="FLOAT_COLOR", domain="POINT")
    color_values = colors_batch.flatten().tolist()
    attribute.data.foreach_set("color", color_values)
    attribute_conf = mesh.attributes.new(name="conf", type="FLOAT", domain="POINT")
    conf_values = conf_batch.tolist()
    attribute_conf.data.foreach_set("value", conf_values)
    obj = bpy.data.objects.new("Points", mesh)

    # Link to the provided collection, or fallback to active collection
    if collection is not None:
        collection.objects.link(obj)
    else:
        bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new(name="PointMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_name = "point_color"
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
    output_node_material = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output_node_material.inputs['Surface'])
    geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    node_group = bpy.data.node_groups.new(name="PointCloud", type='GeometryNodeTree')
    node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket(name="Threshold", in_out="INPUT", socket_type="NodeSocketFloat")
    node_group.interface.items_tree[-1].default_value = 0.5
    node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    geo_mod.node_group = node_group
    input_node = node_group.nodes.new('NodeGroupInput')
    output_node = node_group.nodes.new('NodeGroupOutput')
    mesh_to_points = node_group.nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points.inputs['Radius'].default_value = 0.002
    named_attr = node_group.nodes.new('GeometryNodeInputNamedAttribute')
    named_attr.inputs['Name'].default_value = "conf"
    named_attr.data_type = 'FLOAT'
    compare = node_group.nodes.new('FunctionNodeCompare')
    compare.data_type = 'FLOAT'
    compare.operation = 'LESS_THAN'
    delete_geo = node_group.nodes.new('GeometryNodeDeleteGeometry')
    delete_geo.domain = 'POINT'
    set_material_node = node_group.nodes.new('GeometryNodeSetMaterial')
    set_material_node.inputs['Material'].default_value = mat
    node_group.links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
    node_group.links.new(mesh_to_points.outputs['Points'], delete_geo.inputs['Geometry'])
    node_group.links.new(named_attr.outputs['Attribute'], compare.inputs['A'])
    node_group.links.new(input_node.outputs['Threshold'], compare.inputs['B'])
    node_group.links.new(compare.outputs['Result'], delete_geo.inputs['Selection'])
    node_group.links.new(delete_geo.outputs['Geometry'], set_material_node.inputs['Geometry'])
    node_group.links.new(set_material_node.outputs['Geometry'], output_node.inputs['Geometry'])
    
def create_cameras(predictions, collection=None, image_width=None, image_height=None):
    scene = bpy.context.scene
    if image_width is None or image_height is None:
        H, W = predictions['images'].shape[1:3]
        image_width = W
        image_height = H
    K0 = predictions["intrinsic"][0]
    pixel_aspect_y = K0[1,1] / K0[0,0]
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(pixel_aspect_y)
    num_cameras = len(predictions["extrinsic"])
    if len(predictions["intrinsic"]) != num_cameras:
        raise ValueError("Extrinsic and intrinsic lists must have the same length")

    # Optional: get image paths from predictions, if available
    image_paths = predictions.get("image_paths", None)

    T = np.diag([1.0, -1.0, -1.0, 1.0])
    for i in range(num_cameras):
        # Name from image file if available
        if image_paths and i < len(image_paths):
            import os
            base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
            cam_name = base_name
        else:
            cam_name = f"Camera_{i}"

        cam_data = bpy.data.cameras.new(name=cam_name)
        K = predictions["intrinsic"][i]
        f_x = K[0,0]
        c_x = K[0,2]
        c_y = K[1,2]
        sensor_width = 36.0
        cam_data.sensor_width = sensor_width
        cam_data.lens = (f_x / image_width) * sensor_width
        cam_data.shift_x = (c_x - image_width / 2.0) / image_width
        cam_data.shift_y = (c_y - image_height / 2.0) / image_height
        cam_obj = bpy.data.objects.new(name=cam_name, object_data=cam_data)

        if collection is not None:
            collection.objects.link(cam_obj)
        else:
            scene.collection.objects.link(cam_obj)
        
        ext = predictions["extrinsic"][i]
        E = np.vstack((ext, [0, 0, 0, 1]))
        E_inv = np.linalg.inv(E)
        M = np.dot(E_inv, T)
        cam_obj.matrix_world = Matrix(M.tolist())
        R = Matrix.Rotation(math.radians(-90), 4, 'X')
        cam_obj.matrix_world = R @ cam_obj.matrix_world
