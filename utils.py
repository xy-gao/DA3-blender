import glob
import os
import numpy as np
import bpy
from mathutils import Matrix
import math
import torch

from depth_anything_3.utils.alignment import (
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
            # least_squares_scale_scalar(target, source) returns scale such that source * scale ≈ target
            # We want curr_depth * scale ≈ prev_depth, so target=all_prev, source=all_curr
            scale = least_squares_scale_scalar(all_prev, all_curr)
        else:
            scale = torch.tensor(1.0) # 1x scale if there were no overlap frames with at least 22 non-sky pixels
            
        scale_val = float(scale.item())
        print(f"Batch {i} alignment: scale={scale_val}")
        
        # Step 1: Scale depth and extrinsic translations together (like DA3 does)
        # This handles all scaling in one place
        curr_pred.depth = _to_numpy(curr_depth * scale)
        curr_ext[:, :, 3] = curr_ext[:, :, 3] * scale  # scale all translations
        
        # Step 2: Compute rigid alignment transform from first overlap frame
        # We want to find T such that: E_curr_scaled @ T ≈ E_prev
        # Rearranging: T = inv(E_curr_scaled) @ E_prev
        E_prev, E_curr_orig = transforms[0]
        E_curr_scaled = _extrinsic_to_4x4_torch(curr_ext[curr_indices.index(common_indices[0])])
        T_align = _invert_4x4_torch(E_curr_scaled) @ E_prev
        
        # Step 3: Apply rigid alignment to all extrinsics
        # E_new = E_curr_scaled @ T
        new_extrinsics = []
        for ext_3x4 in curr_ext:
            E_curr = _extrinsic_to_4x4_torch(ext_3x4)
            E_new = E_curr @ T_align
            new_extrinsics.append(E_new[:3, :4])
            
        curr_pred.extrinsics = _to_numpy(torch.stack(new_extrinsics))
        
        # Add the aligned prediction for this batch to the result list
        aligned_predictions.append(curr_pred)
        prev_pred = curr_pred
        prev_indices = curr_indices
        
    # We've finished all batches, return a list of aligned predictions
    return aligned_predictions

def compute_motion_scores(predictions, threshold_ratio=0.1):
    """
    Computes a motion score for each pixel based on consistency with other frames.
    Score is the number of other frames that see 'empty space' where the point should be.
    """
    import torch
    
    # Collect all data
    all_depths = []
    all_extrinsics = []
    all_intrinsics = []
    frame_mapping = [] # List of (batch_index, frame_index_in_batch)
    
    for b_idx, pred in enumerate(predictions):
        # Ensure we have tensors
        d = _to_tensor(pred.depth).float()
        e = _to_tensor(pred.extrinsics).float()
        k = _to_tensor(pred.intrinsics).float()
        
        # Initialize motion attribute on prediction object
        if not hasattr(pred, 'motion'):
            pred.motion = torch.zeros_like(d)
            
        for f_idx in range(d.shape[0]):
            all_depths.append(d[f_idx])
            all_extrinsics.append(e[f_idx])
            all_intrinsics.append(k[f_idx])
            frame_mapping.append((b_idx, f_idx))
            
    if not all_depths:
        return

    # Stack
    depths = torch.stack(all_depths) # [N, H, W]
    extrinsics = torch.stack(all_extrinsics) # [N, 3, 4]
    intrinsics = torch.stack(all_intrinsics) # [N, 3, 3]
    
    N, H, W = depths.shape
    device = depths.device
    
    print(f"Computing motion scores for {N} frames...")
    
    # Construct 4x4 matrices
    Es = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    Es[:, :3, :4] = extrinsics
    Es_inv = torch.linalg.inv(Es)
    
    # Pixel grid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels_hom = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0).float() # [3, HW]
    
    # Loop over source frames
    for i in range(N):
        if i % 10 == 0:
            print(f"  Processing frame {i+1}/{N}")
            
        # Unproject frame i
        K_i_inv = torch.linalg.inv(intrinsics[i])
        rays_i = K_i_inv @ pixels_hom # [3, HW]
        d_i = depths[i].flatten() # [HW]
        
        # Filter valid depth
        valid_mask = d_i > 0
        if not valid_mask.any():
            continue
            
        points_cam_i = rays_i[:, valid_mask] * d_i[valid_mask].unsqueeze(0) # [3, M]
        points_cam_i_hom = torch.cat([points_cam_i, torch.ones((1, points_cam_i.shape[1]), device=device)], dim=0) # [4, M]
        
        # Transform to world
        points_world_hom = Es_inv[i] @ points_cam_i_hom # [4, M]
        
        motion_votes = torch.zeros(points_cam_i.shape[1], device=device)
        
        # Check against all other frames j
        # Optimization: Process in chunks if N is large? 
        # For now, simple loop.
        for j in range(N):
            if i == j:
                continue
                
            # Project to frame j
            points_cam_j_hom = Es[j] @ points_world_hom # [4, M]
            # Check if in front of camera
            z_j = points_cam_j_hom[2]
            in_front = z_j > 0.1 # Near plane
            
            if not in_front.any():
                continue
                
            # Project to pixels
            points_cam_j = points_cam_j_hom[:3]
            proj_j = intrinsics[j] @ points_cam_j
            u_j = proj_j[0] / proj_j[2]
            v_j = proj_j[1] / proj_j[2]
            
            # Check bounds
            in_bounds = (u_j >= 0) & (u_j < W - 1) & (v_j >= 0) & (v_j < H - 1) & in_front
            
            if not in_bounds.any():
                continue
                
            # Sample depth from frame j
            u_j_int = torch.round(u_j).long()
            v_j_int = torch.round(v_j).long()
            
            # Filter indices
            valid_indices = torch.where(in_bounds)[0]
            
            u_sample = u_j_int[valid_indices]
            v_sample = v_j_int[valid_indices]
            
            d_target = depths[j, v_sample, u_sample]
            d_proj = z_j[valid_indices]
            
            # Check for "empty space"
            # If d_target > d_proj * (1 + threshold)
            is_empty = d_target > d_proj * (1 + threshold_ratio)
            
            # Accumulate votes
            motion_votes[valid_indices[is_empty]] += 1
            
        # Store result
        full_motion = torch.zeros(H*W, device=device)
        full_motion[valid_mask] = motion_votes
        
        # Save to prediction object
        b_idx, f_idx = frame_mapping[i]
        predictions[b_idx].motion[f_idx] = full_motion.reshape(H, W)

def convert_prediction_to_dict(prediction, image_paths=None, output_debug_images=False):
    predictions = {}

    # images is already numpy in your current pipeline
    predictions['images'] = prediction.processed_images.astype(np.float32) / 255.0  # [N, H, W, 3]

    # depth / extrinsics / intrinsics may be torch tensors after combination; ensure numpy
    predictions['depth'] = _to_numpy(prediction.depth)
    predictions['extrinsic'] = _to_numpy(prediction.extrinsics)
    predictions['intrinsic'] = _to_numpy(prediction.intrinsics)
    predictions['conf'] = _to_numpy(prediction.conf)
    
    if hasattr(prediction, 'motion'):
        predictions['motion'] = _to_numpy(prediction.motion)

    if image_paths is not None and output_debug_images:
        predictions['image_paths'] = image_paths
        
        # Save debug images
        try:
            import cv2
            # Create debug directory
            first_img_dir = os.path.dirname(image_paths[0])
            debug_dir = os.path.join(first_img_dir, "debug_output")
            os.makedirs(debug_dir, exist_ok=True)
            
            for i, img_path in enumerate(image_paths):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Depth
                depth_map = predictions['depth'][i]
                # Normalize depth for visualization: 0-255
                d_min = np.nanmin(depth_map)
                d_max = np.nanmax(depth_map)
                if d_max > d_min:
                    depth_norm = ((depth_map - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
                
                depth_filename = os.path.join(debug_dir, f"{base_name}_depth.png")
                cv2.imwrite(depth_filename, depth_norm)
                
                # Confidence
                conf_map = predictions['conf'][i]
                # Scale confidence: * 10, clip to 255
                conf_scaled = np.clip(conf_map * 10.0, 0, 255).astype(np.uint8)
                
                conf_filename = os.path.join(debug_dir, f"{base_name}_conf.png")
                cv2.imwrite(conf_filename, conf_scaled)

                # Color Image
                color_img = predictions['images'][i]
                color_img_uint8 = (np.clip(color_img, 0, 1) * 255).astype(np.uint8)
                color_img_bgr = cv2.cvtColor(color_img_uint8, cv2.COLOR_RGB2BGR)
                color_filename = os.path.join(debug_dir, f"{base_name}_color.png")
                cv2.imwrite(color_filename, color_img_bgr)

                # Bad Confidence Overlay
                H, W = conf_map.shape
                bad_img = np.zeros((H, W, 4), dtype=np.uint8) # BGRA
                
                # Yellow for conf <= 2.0
                mask_yellow = (conf_map <= 2.0)
                bad_img[mask_yellow] = [0, 255, 255, 255] # Yellow
                
                # Red for conf <= 1.0
                mask_red = (conf_map <= 1.0)
                bad_img[mask_red] = [0, 0, 255, 255] # Red
                
                # Magenta for conf <= 1.0 adjacent to conf > 1.0
                mask_good = (conf_map > 1.0)
                kernel = np.ones((3,3), np.uint8)
                # Dilate good area to find neighbors
                dilated_good = cv2.dilate(mask_good.astype(np.uint8), kernel, iterations=1).astype(bool)
                # Intersection: Is red AND is touched by good
                mask_magenta = mask_red & dilated_good
                bad_img[mask_magenta] = [255, 0, 255, 255] # Magenta
                
                bad_filename = os.path.join(debug_dir, f"{base_name}_bad.png")
                cv2.imwrite(bad_filename, bad_img)

                # Depth Gradient
                grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                g_min = np.nanmin(grad_mag)
                g_max = np.nanmax(grad_mag)
                if g_max > g_min:
                    grad_norm = ((grad_mag - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
                else:
                    grad_norm = np.zeros_like(grad_mag, dtype=np.uint8)
                
                grad_filename = os.path.join(debug_dir, f"{base_name}_grad.png")
                cv2.imwrite(grad_filename, grad_norm)
                
        except ImportError:
            print("Warning: cv2 not found, skipping debug image output.")
        except Exception as e:
            print(f"Warning: Failed to save debug images: {e}")
    elif image_paths is not None:
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

# Based on da3_repo/src/depth_anything_3/model/da3.py
def combine_base_and_metric(base_list, metric_list):
    """Combine base predictions (with poses) with metric predictions (no poses).

    This version operates purely on [N, H, W] tensors per batch and
    re-implements the metric scaling logic from DA3 so that batches may
    have different sizes (e.g. a shorter last batch).

    Args:
        base_list:   list of base `Prediction` objects (one per batch), each with
                     depth [N_b, H, W], conf [N_b, H, W], intrinsics [N_b, 3, 3],
                     extrinsics [N_b, 3, 4].
        metric_list: list of metric `Prediction` objects (one per batch), each with
                     depth [N_m, H, W], sky [N_m, H, W]. For scale_base you typically
                     pass a single-element list and let total metric frames
                     be <= total base frames.

    Returns:
        List of base `Prediction` objects (same length as base_list) whose
        depths and extrinsics have been globally scaled to metric units.
    """

    if not base_list:
        return []

    # Concatenate all base frames into a single [Nb_total, H, W]
    base_depth_all = []
    base_conf_all = []
    base_intr_all = []

    for pred in base_list:
        d = _to_tensor(pred.depth).float()      # [N_b, H, W]
        c = _to_tensor(pred.conf).float()       # [N_b, H, W]
        K = _to_tensor(pred.intrinsics).float() # [N_b, 3, 3]
        if d.ndim != 3 or c.ndim != 3:
            raise ValueError(f"Base depth/conf must be [N,H,W], got depth={d.shape}, conf={c.shape}")
        base_depth_all.append(d)
        base_conf_all.append(c)
        base_intr_all.append(K)

    depth_all = torch.cat(base_depth_all, dim=0)   # [Nb_total, H, W]
    conf_all = torch.cat(base_conf_all, dim=0)     # [Nb_total, H, W]
    intr_all = torch.cat(base_intr_all, dim=0)     # [Nb_total, 3, 3]

    # Concatenate all metric frames similarly
    metric_depth_all = []
    sky_all = []
    for pred in metric_list:
        md = _to_tensor(pred.depth).float()   # [Nm, H, W]
        sky = _to_tensor(pred.sky).float()    # [Nm, H, W]
        if md.ndim != 3 or sky.ndim != 3:
            raise ValueError(f"Metric depth/sky must be [N,H,W], got depth={md.shape}, sky={sky.shape}")
        metric_depth_all.append(md)
        sky_all.append(sky)

    if not metric_depth_all:
        raise ValueError("Metric prediction list is empty or missing required fields")

    metric_all = torch.cat(metric_depth_all, dim=0)   # [Nm_total, H, W]
    sky_all = torch.cat(sky_all, dim=0)               # [Nm_total, H, W]

    Nb_total = depth_all.shape[0]
    Nm_total = metric_all.shape[0]

    # Restrict to overlapping frames in the sequence sense
    N_overlap = min(Nb_total, Nm_total)
    if N_overlap <= 0:
        raise ValueError("Metric prediction has no frames; cannot compute scale.")

    depth_overlap = depth_all[:N_overlap]        # [N_overlap, H, W]
    metric_overlap = metric_all[:N_overlap]      # [N_overlap, H, W]
    sky_overlap = sky_all[:N_overlap]            # [N_overlap, H, W]
    ixt_overlap = intr_all[:N_overlap]           # [N_overlap, 3, 3]

    # Inline metric scaling logic from DA3's apply_metric_scaling for [N, H, W]
    # focal_length = (fx + fy) / 2, depth_scaled = depth * (f / scale_factor)
    scale_factor_metric = 300.0
    focal_length = (ixt_overlap[:, 0, 0] + ixt_overlap[:, 1, 1]) / 2.0   # [N_overlap]
    metric_scaled = metric_overlap * (focal_length[:, None, None] / scale_factor_metric)

    # Non-sky mask and alignment only on overlapping frames
    non_sky_mask = compute_sky_mask(sky_overlap, threshold=0.3)  # [N_overlap, H, W]
    if non_sky_mask.sum() <= 10:
        raise ValueError("Insufficient non-sky pixels for alignment")

    depth_conf_overlap = conf_all[:N_overlap]   # [N_overlap, H, W]
    depth_conf_ns = depth_conf_overlap[non_sky_mask]
    depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
    median_conf = torch.quantile(depth_conf_sampled, 0.5)

    align_mask = compute_alignment_mask(
        depth_conf_overlap, non_sky_mask, depth_overlap, metric_scaled, median_conf
    )

    valid_depth = depth_overlap[align_mask]
    valid_metric_depth = metric_scaled[align_mask]
    scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

    # Scale depth and extrinsics for each base batch
    scaled_base_list = []
    for pred in base_list:
        ext = _to_tensor(pred.extrinsics)
        if ext is not None:
            if ext.ndim != 3 or ext.shape[1:] != (3, 4):
                raise ValueError(f"Expected extrinsics [N,3,4], got {ext.shape}")
            ext = ext.float()
            ext[:, :, 3] *= scale_factor

        pred.depth = _to_tensor(pred.depth) * scale_factor
        if ext is not None:
            pred.extrinsics = ext
        pred.is_metric = 1
        pred.scale_factor = float(scale_factor.item())
        scaled_base_list.append(pred)

    return scaled_base_list


def combine_base_with_metric_depth(base, metric):
    """Combine base prediction cameras with raw metric model depth.

    This variant keeps **base intrinsics/extrinsics/conf** but **replaces
    depth with metric.depth in metres**, then applies the same sky-handling
    logic as `combine_base_and_metric`.

    Assumes shapes:
      - base.depth:        [N, H, W]
      - metric.depth:      [N, H, W]
      - base.intrinsics:   [N, 3, 3]
      - base.extrinsics:   [N, 3, 4]
      - metric.sky:        [N, H, W]
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
    # Use robust alignment mask - convert conf to tensor if needed
    depth_conf = _to_tensor(output.conf).float()
    depth_conf_ns = depth_conf[non_sky_mask]
    depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
    median_conf = torch.quantile(depth_conf_sampled, 0.5)

    align_mask = compute_alignment_mask(
        depth_conf, non_sky_mask, base_depth, metric_depth, median_conf
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

    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsics [N,3,4], got {extrinsics.shape}")

    extrinsics = extrinsics.float()
    extrinsics[:, :, 3] = extrinsics[:, :, 3] * scale_factor

    # Write back into output: metric depth + scaled base cameras
    output.depth = depth
    output.extrinsics = extrinsics
    output.is_metric = 1
    output.scale_factor = float(scale_factor.item())

    return output

def get_or_create_point_material():
    mat = bpy.data.materials.get("PointMaterial")
    if mat is None:
        mat = bpy.data.materials.new(name="PointMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)
        
        # Image color attribute
        attr_node = nodes.new('ShaderNodeAttribute')
        attr_node.attribute_name = "point_color"
        attr_node.location = (-600, 200)
        
        # Confidence attribute (raw values)
        conf_attr_node = nodes.new('ShaderNodeAttribute')
        conf_attr_node.attribute_name = "conf"
        conf_attr_node.location = (-600, -200)
        
        # Map Range: 0-10 -> 0-1 (so conf values map to reasonable ramp positions)
        map_range = nodes.new('ShaderNodeMapRange')
        map_range.location = (-400, -200)
        map_range.clamp = True
        map_range.inputs['From Min'].default_value = 0.0
        map_range.inputs['From Max'].default_value = 10.0
        map_range.inputs['To Min'].default_value = 0.0
        map_range.inputs['To Max'].default_value = 1.0
        
        # Color Ramp: red (low) -> green (mid) -> blue (high)
        # Positions: 0.2 = conf 2, 0.5 = conf 5, 0.6 = conf 6
        color_ramp = nodes.new('ShaderNodeValToRGB')
        color_ramp.location = (-150, -200)
        # Clear default elements and set up: red at 0, green at 0.5-0.6, blue at 1
        ramp = color_ramp.color_ramp
        ramp.elements[0].position = 0.0
        ramp.elements[0].color = (1, 0, 0, 1)  # Red (conf < 2)
        ramp.elements[1].position = 0.2
        ramp.elements[1].color = (1, 0, 0, 1)  # Still red at conf=2
        # Add green zone
        green_start = ramp.elements.new(0.5)
        green_start.color = (0, 1, 0, 1)  # Green at conf=5
        green_end = ramp.elements.new(0.6)
        green_end.color = (0, 1, 0, 1)  # Green at conf=6
        # Add blue
        blue_elem = ramp.elements.new(1.0)
        blue_elem.color = (0, 0, 1, 1)  # Blue at conf=10
        
        # Mix shader to switch between image color and confidence color
        mix_node = nodes.new('ShaderNodeMix')
        mix_node.data_type = 'RGBA'
        mix_node.location = (100, 100)
        mix_node.inputs['Factor'].default_value = 0.0  # 0 = image color, 1 = confidence color
        
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (300, 100)
        
        output_node_material = nodes.new('ShaderNodeOutputMaterial')
        output_node_material.location = (550, 100)
        
        # Connect nodes
        links.new(conf_attr_node.outputs['Fac'], map_range.inputs['Value'])
        links.new(map_range.outputs['Result'], color_ramp.inputs['Fac'])
        links.new(attr_node.outputs['Color'], mix_node.inputs['A'])
        links.new(color_ramp.outputs['Color'], mix_node.inputs['B'])
        links.new(mix_node.outputs['Result'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output_node_material.inputs['Surface'])
    return mat

def import_point_cloud(d, collection=None, filter_edges=True, min_confidence=0.5):
    points = d["world_points_from_depth"]
    images = d["images"]
    conf = d["conf"]

    # Filter confidence based on depth gradient
    if filter_edges and "depth" in d:
        try:
            import cv2
            depth = d["depth"]
            for i in range(len(depth)):
                dm = depth[i]
                gx = cv2.Sobel(dm, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(dm, cv2.CV_64F, 0, 1, ksize=3)
                mag = np.sqrt(gx**2 + gy**2)
                mn, mx = np.nanmin(mag), np.nanmax(mag)
                if mx > mn:
                    norm = (mag - mn) / (mx - mn)
                else:
                    norm = np.zeros_like(mag)
                
                # Set confidence to 0 if normalized gradient >= 12/255
                mask = norm >= (12.0 / 255.0)
                conf[i][mask] = 0.0
        except Exception as e:
            print(f"Failed to filter confidence by gradient: {e}")

    points_batch = points.reshape(-1, 3)
    reordered_points_batch = points_batch.copy()
    reordered_points_batch[:, [0, 1, 2]] = points_batch[:, [0, 2, 1]]
    reordered_points_batch[:, 2] = -reordered_points_batch[:, 2]
    points_batch = reordered_points_batch
    colors_batch = images.reshape(-1, 3)
    colors_batch = np.hstack((colors_batch, np.ones((colors_batch.shape[0], 1))))
    conf_batch = conf.reshape(-1)
    
    motion_batch = None
    if 'motion' in d:
        motion_batch = d['motion'].reshape(-1)

    # Remove points with low confidence
    if min_confidence > 0:
        valid_mask = conf_batch >= min_confidence
        points_batch = points_batch[valid_mask]
        colors_batch = colors_batch[valid_mask]
        conf_batch = conf_batch[valid_mask]
        if motion_batch is not None:
            motion_batch = motion_batch[valid_mask]
    
    if len(conf_batch) > 0:
        print(f"DEBUG confidence: min={conf_batch.min():.4f}, max={conf_batch.max():.4f}")
    
    mesh = bpy.data.meshes.new(name="Points")
    vertices = points_batch.tolist()
    mesh.from_pydata(vertices, [], [])
    
    # Image colors
    attribute = mesh.attributes.new(name="point_color", type="FLOAT_COLOR", domain="POINT")
    color_values = colors_batch.flatten().tolist()
    attribute.data.foreach_set("color", color_values)
    
    # Raw confidence value
    attribute_conf = mesh.attributes.new(name="conf", type="FLOAT", domain="POINT")
    conf_values = conf_batch.tolist()
    attribute_conf.data.foreach_set("value", conf_values)
    
    # Motion score
    if motion_batch is not None:
        attribute_motion = mesh.attributes.new(name="motion", type="FLOAT", domain="POINT")
        motion_values = motion_batch.tolist()
        attribute_motion.data.foreach_set("value", motion_values)
    
    obj = bpy.data.objects.new("Points", mesh)

    # Link to the provided collection, or fallback to active collection
    if collection is not None:
        collection.objects.link(obj)
    else:
        bpy.context.collection.objects.link(obj)

    # Reuse existing PointMaterial or create new one
    mat = get_or_create_point_material()
    
    # Add material to object so it shows up in Shading mode
    obj.data.materials.append(mat)
    
    # Geometry nodes setup
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

def create_image_material(image_path):
    name = os.path.basename(image_path)
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        tex_coord = nodes.new('ShaderNodeTexCoord')
        tex_coord.location = (-800, 200)
        
        tex_image = nodes.new('ShaderNodeTexImage')
        tex_image.location = (-500, 200)
        try:
            # Check if image is already loaded
            img_name = os.path.basename(image_path)
            img = bpy.data.images.get(img_name)
            if img is None:
                img = bpy.data.images.load(image_path)
            tex_image.image = img
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (-200, 200)
        bsdf.inputs['Roughness'].default_value = 1.0
        # Try to set specular to 0 to avoid shiny photos
        if 'Specular IOR Level' in bsdf.inputs:
             bsdf.inputs['Specular IOR Level'].default_value = 0.0
        elif 'Specular' in bsdf.inputs:
             bsdf.inputs['Specular'].default_value = 0.0
        
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (100, 200)
        
        links.new(tex_coord.outputs['UV'], tex_image.inputs['Vector'])
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat

def import_mesh_from_depth(d, collection=None, filter_edges=True, min_confidence=0.5):
    points = d["world_points_from_depth"] # [N, H, W, 3]
    images = d["images"] # [N, H, W, 3]
    conf = d["conf"] # [N, H, W]

    # Filter confidence based on depth gradient (Same as import_point_cloud)
    if filter_edges and "depth" in d:
        try:
            import cv2
            depth = d["depth"]
            for i in range(len(depth)):
                dm = depth[i]
                gx = cv2.Sobel(dm, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(dm, cv2.CV_64F, 0, 1, ksize=3)
                mag = np.sqrt(gx**2 + gy**2)
                mn, mx = np.nanmin(mag), np.nanmax(mag)
                if mx > mn:
                    norm = (mag - mn) / (mx - mn)
                else:
                    norm = np.zeros_like(mag)
                
                # Set confidence to 0 if normalized gradient >= 12/255
                mask = norm >= (12.0 / 255.0)
                conf[i][mask] = 0.0
        except Exception as e:
            print(f"Failed to filter confidence by gradient: {e}")
    
    N, H, W, _ = points.shape
    
    # Generate grid faces once (shared for all images in batch)
    # Grid indices: (r, c) -> r*W + c
    # Quad: (r, c), (r, c+1), (r+1, c+1), (r+1, c)
    r = np.arange(H - 1)
    c = np.arange(W - 1)
    rr, cc = np.meshgrid(r, c, indexing='ij')
    v0 = rr * W + cc
    v1 = rr * W + (cc + 1)
    v2 = (rr + 1) * W + (cc + 1)
    v3 = (rr + 1) * W + cc
    # Blender expects counter-clockwise winding for front faces
    faces = np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)
    
    # Generate UVs once
    u_coords = np.linspace(0, 1, W, dtype=np.float32)
    v_coords = np.linspace(1, 0, H, dtype=np.float32) # Top is 1, Bottom is 0
    uu, vv = np.meshgrid(u_coords, v_coords)
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

    for i in range(N):
        # Prepare data for this image
        pts = points[i].reshape(-1, 3)
        # Apply the same coordinate transform as import_point_cloud
        pts_transformed = pts.copy()
        pts_transformed[:, [0, 1, 2]] = pts[:, [0, 2, 1]]
        pts_transformed[:, 2] = -pts_transformed[:, 2]
        
        cols = images[i].reshape(-1, 3)
        cols = np.hstack((cols, np.ones((cols.shape[0], 1)))) # RGBA
        confs = conf[i].reshape(-1)
        
        motion_vals = None
        if 'motion' in d:
            motion_vals = d['motion'][i].reshape(-1)
        
        # Create Mesh
        if "image_paths" in d and i < len(d["image_paths"]):
            base_name = os.path.splitext(os.path.basename(d["image_paths"][i]))[0]
            mesh_name = f"Mesh_{base_name}"
        else:
            mesh_name = f"Mesh_Img_{i}"
            
        mesh = bpy.data.meshes.new(name=mesh_name)
        mesh.from_pydata(pts_transformed.tolist(), [], faces.tolist())
        
        # Add UVs
        uv_layer = mesh.uv_layers.new(name="UVMap")
        loop_vert_indices = np.zeros(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vert_indices)
        loop_uvs = uvs[loop_vert_indices]
        uv_layer.data.foreach_set("uv", loop_uvs.flatten())
        
        # Add Attributes
        # Color
        col_attr = mesh.attributes.new(name="point_color", type="FLOAT_COLOR", domain="POINT")
        col_attr.data.foreach_set("color", cols.flatten())
        
        # Confidence
        conf_attr = mesh.attributes.new(name="conf", type="FLOAT", domain="POINT")
        conf_attr.data.foreach_set("value", confs)
        
        # Motion
        if motion_vals is not None:
            motion_attr = mesh.attributes.new(name="motion", type="FLOAT", domain="POINT")
            motion_attr.data.foreach_set("value", motion_vals)
        
        obj = bpy.data.objects.new(mesh_name, mesh)
        if collection:
            collection.objects.link(obj)
        else:
            bpy.context.collection.objects.link(obj)
            
        # Add Material
        if "image_paths" in d:
            img_path = d["image_paths"][i]
            mat = create_image_material(img_path)
        else:
            mat = get_or_create_point_material()
        obj.data.materials.append(mat)
            
        # Add Geometry Nodes to filter stretched edges and low confidence
        if filter_edges:
            mod = obj.modifiers.new(name="FilterMesh", type='NODES')
            group_name = "FilterDepthMesh"
            group = bpy.data.node_groups.get(group_name)
            if not group:
                group = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
                group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
                group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
                
                # Nodes
                in_node = group.nodes.new('NodeGroupInput')
                out_node = group.nodes.new('NodeGroupOutput')
                
                # 1. Filter by Confidence (Delete Points)
                del_conf = group.nodes.new('GeometryNodeDeleteGeometry')
                del_conf.domain = 'POINT'
                named_attr = group.nodes.new('GeometryNodeInputNamedAttribute')
                named_attr.data_type = 'FLOAT'
                named_attr.inputs['Name'].default_value = "conf"
                compare_conf = group.nodes.new('FunctionNodeCompare')
                compare_conf.operation = 'LESS_THAN'
                compare_conf.inputs['B'].default_value = min_confidence
                
                group.links.new(named_attr.outputs['Attribute'], compare_conf.inputs['A'])
                group.links.new(compare_conf.outputs['Result'], del_conf.inputs['Selection'])
                
                # 2. Filter by Edge Length (Delete Edges)
                del_edge = group.nodes.new('GeometryNodeDeleteGeometry')
                del_edge.domain = 'EDGE'
                
                # Calculate Edge Length manually (Edge Length node name varies)
                edge_verts = group.nodes.new('GeometryNodeInputMeshEdgeVertices')
                pos = group.nodes.new('GeometryNodeInputPosition')
                
                sample_pos1 = group.nodes.new('GeometryNodeSampleIndex')
                sample_pos1.data_type = 'FLOAT_VECTOR'
                sample_pos1.domain = 'POINT'
                
                sample_pos2 = group.nodes.new('GeometryNodeSampleIndex')
                sample_pos2.data_type = 'FLOAT_VECTOR'
                sample_pos2.domain = 'POINT'
                
                dist = group.nodes.new('ShaderNodeVectorMath')
                dist.operation = 'DISTANCE'
                
                compare_edge = group.nodes.new('FunctionNodeCompare')
                compare_edge.operation = 'GREATER_THAN'
                compare_edge.inputs['B'].default_value = 0.1 # Threshold for jump (meters)
                
                # Connect Geometry (from del_conf)
                group.links.new(del_conf.outputs['Geometry'], sample_pos1.inputs['Geometry'])
                group.links.new(del_conf.outputs['Geometry'], sample_pos2.inputs['Geometry'])
                
                # Connect Indices and Values
                group.links.new(edge_verts.outputs['Vertex Index 1'], sample_pos1.inputs['Index'])
                group.links.new(pos.outputs['Position'], sample_pos1.inputs['Value'])
                
                group.links.new(edge_verts.outputs['Vertex Index 2'], sample_pos2.inputs['Index'])
                group.links.new(pos.outputs['Position'], sample_pos2.inputs['Value'])
                
                # Calculate Distance
                group.links.new(sample_pos1.outputs['Value'], dist.inputs[0])
                group.links.new(sample_pos2.outputs['Value'], dist.inputs[1])
                
                # Compare
                group.links.new(dist.outputs['Value'], compare_edge.inputs['A'])
                group.links.new(compare_edge.outputs['Result'], del_edge.inputs['Selection'])
                
                # Connect Main Flow
                group.links.new(in_node.outputs['Geometry'], del_conf.inputs['Geometry'])
                group.links.new(del_conf.outputs['Geometry'], del_edge.inputs['Geometry'])
                group.links.new(del_edge.outputs['Geometry'], out_node.inputs['Geometry'])
                
            mod.node_group = group

