#!/usr/bin/env python3
"""
Script to smooth and interpolate 2D egocentric hand poses from MediaPipe detection.
Similar to smooth_hand3d.py but adapted for 2D keypoints.

Usage: python smooth_hand2d.py --groups Attendings --subj_name 09_30 --camera Ego
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hand skeleton edges for 21-joint topology (wrist=0)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def plot_hand_2d(ax, pts, color, label=None):
    """Plot 2D hand keypoints and connections.
    
    Args:
        ax: matplotlib axis
        pts: [21, 3] array with x, y, confidence
        color: color for plotting
        label: optional label for legend
    """
    conf = pts[:, 2] > 0
    xy = pts[:, :2]
    
    # Plot joints
    ax.scatter(xy[conf, 0], xy[conf, 1], c=color, s=20, label=label, alpha=0.8)
    
    # Plot connections
    for i, j in HAND_EDGES:
        if conf[i] and conf[j]:
            xs = [xy[i, 0], xy[j, 0]]
            ys = [xy[i, 1], xy[j, 1]]
            ax.plot(xs, ys, color=color, linewidth=1, alpha=0.7)

def plot_hand_2d_overlay(ax, pts, color, label=None, linewidth=3, pointsize=40):
    """Plot 2D hand keypoints and connections optimized for overlay on images.
    
    Args:
        ax: matplotlib axis
        pts: [21, 3] array with x, y, confidence
        color: color for plotting
        label: optional label for legend
        linewidth: thickness of connection lines
        pointsize: size of joint markers
    """
    conf = pts[:, 2] > 0
    xy = pts[:, :2]
    
    # Plot connections first (so they appear behind joints)
    for i, j in HAND_EDGES:
        if conf[i] and conf[j]:
            xs = [xy[i, 0], xy[j, 0]]
            ys = [xy[i, 1], xy[j, 1]]
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.9, solid_capstyle='round')
    
    # Plot joints on top with black outline for better visibility
    if conf.any():
        # Black outline
        ax.scatter(xy[conf, 0], xy[conf, 1], c='black', s=pointsize+10, alpha=0.8, edgecolors='none')
        # Colored center
        ax.scatter(xy[conf, 0], xy[conf, 1], c=color, s=pointsize, label=label, alpha=0.9, edgecolors='black', linewidths=1)

def visualize_2d_comparison(out_data, original_data, out_dir, base_dir, camera, img_size=(1920, 1080)):
    """Visualize comparison between original and smoothed 2D hand poses.
    
    Args:
        out_data: dict with smoothed 'left', 'right', 'image' keys
        original_data: dict with original 'left', 'right', 'image' keys
        out_dir: output directory for visualization
        base_dir: base directory containing Images folder
        camera: camera name (e.g., 'Ego')
        img_size: (width, height) for fallback when no image found
    """
    os.makedirs(out_dir, exist_ok=True)
    left_smooth = out_data['left']
    right_smooth = out_data['right']
    left_orig = original_data['left']
    right_orig = original_data['right']
    names = out_data.get('image', [])
    T = min(len(left_smooth), len(right_smooth), len(left_orig), len(right_orig))
    
    # Path to original images
    img_dir = os.path.join(base_dir, 'Images', camera)
    
    for t in tqdm(range(T), desc='Render pose comparison on images'):
        L_smooth = left_smooth[t]
        R_smooth = right_smooth[t]
        L_orig = left_orig[t]
        R_orig = right_orig[t]
        
        # Get image name
        img_name = names[t] if t < len(names) else f'{t:06d}.jpg'
        
        # Try to load the original image
        img_path = os.path.join(img_dir, img_name)
        background_img = None
        
        if os.path.exists(img_path):
            try:
                background_img = cv2.imread(img_path)
                if background_img is not None:
                    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
                    actual_img_size = (background_img.shape[1], background_img.shape[0])
                else:
                    actual_img_size = img_size
            except Exception as e:
                background_img = None
                actual_img_size = img_size
        else:
            actual_img_size = img_size
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Display background image on both subplots
        for ax in [ax1, ax2]:
            if background_img is not None:
                ax.imshow(background_img)
            else:
                ax.set_facecolor('black')
            ax.set_xlim(0, actual_img_size[0])
            ax.set_ylim(actual_img_size[1], 0)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Left subplot: Original poses
        plot_hand_2d_overlay(ax1, L_orig, color='lightblue', label='Left (Original)', linewidth=2, pointsize=30)
        plot_hand_2d_overlay(ax1, R_orig, color='lightyellow', label='Right (Original)', linewidth=2, pointsize=30)
        ax1.set_title('Original Poses', color='white', fontweight='bold', fontsize=16, pad=20)
        legend1 = ax1.legend(loc='upper right', framealpha=0.8)
        for text in legend1.get_texts():
            text.set_color('white')
            text.set_fontweight('bold')
        
        # Right subplot: Smoothed poses with confidence indication
        plot_hand_2d_confidence(ax2, L_smooth, 'Left', linewidth=3, pointsize=40)
        plot_hand_2d_confidence(ax2, R_smooth, 'Right', linewidth=3, pointsize=40)
        ax2.set_title('Smoothed Poses (Cyan=Original, Yellow=Interpolated)', color='white', fontweight='bold', fontsize=16, pad=20)
        
        fig.suptitle(f'Pose Comparison - {img_name}', color='white', fontweight='bold', fontsize=18)
        
        out_name = os.path.splitext(img_name)[0] + '_comparison.png'
        out_path = os.path.join(out_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

def plot_hand_2d_confidence(ax, pts, hand_label, linewidth=3, pointsize=40):
    """Plot 2D hand with different colors for original vs interpolated keypoints.
    
    Args:
        ax: matplotlib axis
        pts: [21, 3] array with x, y, confidence
        hand_label: 'Left' or 'Right'
        linewidth: thickness of connection lines
        pointsize: size of joint markers
    """
    xy = pts[:, :2]
    conf = pts[:, 2]
    
    # Original poses (conf = 1.0), Interpolated poses (conf = 0.5)
    original_mask = conf >= 0.99  # Original detections
    interp_mask = (conf > 0.1) & (conf < 0.99)  # Interpolated
    
    # Colors based on confidence
    original_color = 'cyan' if hand_label == 'Left' else 'yellow'
    interp_color = 'orange' if hand_label == 'Left' else 'red'
    
    # Plot connections
    for i, j in HAND_EDGES:
        if conf[i] > 0 and conf[j] > 0:
            xs = [xy[i, 0], xy[j, 0]]
            ys = [xy[i, 1], xy[j, 1]]
            # Use interpolated color if either point is interpolated
            if interp_mask[i] or interp_mask[j]:
                color = interp_color
                alpha = 0.7
            else:
                color = original_color
                alpha = 0.9
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')
    
    # Plot joints
    if original_mask.any():
        ax.scatter(xy[original_mask, 0], xy[original_mask, 1], c='black', s=pointsize+10, alpha=0.8)
        ax.scatter(xy[original_mask, 0], xy[original_mask, 1], c=original_color, s=pointsize, 
                  alpha=0.9, edgecolors='black', linewidths=1, label=f'{hand_label} (Original)')
    
    if interp_mask.any():
        ax.scatter(xy[interp_mask, 0], xy[interp_mask, 1], c='black', s=pointsize+10, alpha=0.8)
        ax.scatter(xy[interp_mask, 0], xy[interp_mask, 1], c=interp_color, s=pointsize, 
                  alpha=0.9, edgecolors='black', linewidths=1, label=f'{hand_label} (Interpolated)')

def visualize_2d_sequence(out_data, out_dir, base_dir, camera, img_size=(1920, 1080)):
    """Visualize smoothed 2D hand poses overlayed on original images.
    
    Args:
        out_data: dict with 'left', 'right', 'image' keys
        out_dir: output directory for visualization
        base_dir: base directory containing Images folder
        camera: camera name (e.g., 'Ego')
        img_size: (width, height) for fallback when no image found
    """
    os.makedirs(out_dir, exist_ok=True)
    left = out_data['left']
    right = out_data['right']
    names = out_data.get('image', [])
    T = min(len(left), len(right))
    
    # Path to original images
    img_dir = os.path.join(base_dir, 'Images', camera)
    
    for t in tqdm(range(T), desc='Render smoothed 2D hand plots on images'):
        L = left[t]
        R = right[t]
        
        # Get image name
        img_name = names[t] if t < len(names) else f'{t:06d}.jpg'
        
        # Try to load the original image
        img_path = os.path.join(img_dir, img_name)
        background_img = None
        
        if os.path.exists(img_path):
            try:
                background_img = cv2.imread(img_path)
                if background_img is not None:
                    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
                    actual_img_size = (background_img.shape[1], background_img.shape[0])  # (width, height)
                else:
                    print(f"Warning: Could not read image {img_path}")
                    actual_img_size = img_size
            except Exception as e:
                print(f"Warning: Error loading image {img_path}: {e}")
                background_img = None
                actual_img_size = img_size
        else:
            print(f"Warning: Image not found {img_path}")
            actual_img_size = img_size
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Display background image if available
        if background_img is not None:
            ax.imshow(background_img)
            actual_img_size = (background_img.shape[1], background_img.shape[0])
        else:
            # Create a black background if no image
            ax.set_facecolor('black')
            actual_img_size = img_size
        
        # Plot hands with thicker lines and larger points for visibility on image
        plot_hand_2d_overlay(ax, L, color='cyan', label='Left', linewidth=3, pointsize=40)
        plot_hand_2d_overlay(ax, R, color='yellow', label='Right', linewidth=3, pointsize=40)
        
        # Set axis properties
        ax.set_xlim(0, actual_img_size[0])
        ax.set_ylim(actual_img_size[1], 0)  # Flip Y axis for image coordinates
        ax.set_aspect('equal')
        ax.axis('off')  # Hide axes for cleaner look
        
        # Add legend with better visibility
        legend = ax.legend(loc='upper right', framealpha=0.8, fancybox=True, shadow=True)
        for text in legend.get_texts():
            text.set_color('white')
            text.set_fontweight('bold')
        
        title = f'Smoothed 2D Hands - {img_name}'
        ax.set_title(title, color='white', fontweight='bold', fontsize=14, pad=20)
        
        out_name = os.path.splitext(img_name)[0] + '_smoothed.png'
        out_path = os.path.join(out_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

def compute_center_2d(poses_xy: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Compute hand centers for 2D poses.
    
    Args:
        poses_xy: [T, J, 2] array of 2D keypoints
        valid_mask: [T, J] boolean mask for valid keypoints
        
    Returns:
        centers: [T, 2] array of hand centers
    """
    T, J, _ = poses_xy.shape
    centers = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        mask_t = valid_mask[t]
        if mask_t.any():
            centers[t] = poses_xy[t, mask_t].mean(axis=0)
        else:
            centers[t] = 0.0
    return centers

def detect_speed_spikes_2d(centers: np.ndarray, speed_thresh: float, max_interval: int = None) -> np.ndarray:
    """Detect unrealistic speed spikes in 2D hand movement.
    
    Args:
        centers: [T, 2] array of hand centers
        speed_thresh: maximum allowed movement in pixels per frame
        max_interval: maximum frames to look back for reference
        
    Returns:
        spikes: [T] boolean array marking spike frames
    """
    T = centers.shape[0]
    spikes = np.zeros(T, dtype=bool)
    
    if T <= 1:
        return spikes
    
    last_valid_idx = 0
    
    for i in range(1, T):
        reference_idx = last_valid_idx
        
        # If max_interval exceeded, use immediate previous frame
        if max_interval is not None and (i - last_valid_idx) > max_interval:
            reference_idx = i - 1
        
        # Compute 2D distance from current frame to reference frame
        delta = np.linalg.norm(centers[i] - centers[reference_idx])
        
        if delta > speed_thresh:
            spikes[i] = True
        else:
            last_valid_idx = i
    
    return spikes

def find_invalid_runs(mask: np.ndarray) -> list:
    """Find continuous runs of invalid frames.
    
    Args:
        mask: [T] boolean array where False means invalid
        
    Returns:
        List of (start, end) tuples for invalid runs (inclusive)
    """
    runs = []
    T = mask.shape[0]
    i = 0
    while i < T:
        if not mask[i]:
            j = i
            while j + 1 < T and not mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs

def interpolate_1d(values: np.ndarray, valid: np.ndarray, max_gap: int, max_edge_extrap: int) -> np.ndarray:
    """Interpolate 1D values with gap length constraints.
    
    Args:
        values: [T] array of values
        valid: [T] boolean mask for valid values
        max_gap: maximum gap length to interpolate
        max_edge_extrap: maximum extrapolation at sequence edges
        
    Returns:
        [T] interpolated array with NaN for unfillable gaps
    """
    T = values.shape[0]
    if valid.sum() == 0:
        return np.full(T, np.nan, dtype=np.float32)
    
    out = values.copy().astype(np.float32)
    
    # Interpolate invalid indices
    idx = np.arange(T)
    valid_idx = idx[valid]
    invalid_idx = idx[~valid]
    
    if len(invalid_idx) > 0:
        interp_values = np.interp(invalid_idx, valid_idx, values[valid])
        out[invalid_idx] = interp_values

    # Re-mask long gaps beyond max_gap
    invalid_runs = find_invalid_runs(valid)
    for s, e in invalid_runs:
        run_len = e - s + 1
        if s == 0:
            # Leading gap: allow up to max_edge_extrap
            if run_len > max_edge_extrap:
                out[: e - max_edge_extrap + 1] = np.nan
        elif e == T - 1:
            # Trailing gap: allow up to max_edge_extrap
            if run_len > max_edge_extrap:
                out[s + max_edge_extrap :] = np.nan
        elif run_len > max_gap:
            # Internal gap too long
            out[s : e + 1] = np.nan
    return out

def moving_average_robust(values: np.ndarray, window: int, delta: float) -> np.ndarray:
    """Apply robust moving average with outlier clipping.
    
    Args:
        values: [T] array with potential NaNs
        window: window size for moving average
        delta: clipping threshold for outlier removal
        
    Returns:
        [T] smoothed array
    """
    T = values.shape[0]
    half = window // 2
    padded = np.pad(values, (half, half), mode='edge')
    smoothed = np.empty(T, dtype=np.float32)
    
    for t in range(T):
        win = padded[t : t + window]
        valid = ~np.isnan(win)
        if not valid.any():
            smoothed[t] = np.nan
            continue
        x = win[valid]
        med = np.median(x)
        resid = x - med
        clipped = med + np.clip(resid, -delta, delta)
        smoothed[t] = clipped.mean()
    return smoothed

def filter_sparse_keypoints_2d(kp_list: list, img_names: list, sparse_threshold: float = 50.0, img_size=(640, 480)) -> np.ndarray:
    """Filter out frames with overly sparse 2D keypoints.
    
    Args:
        kp_list: list of [J, 3] arrays (x, y, conf) length T
        img_names: list of image names for logging
        sparse_threshold: threshold for average distance to centroid in pixels
        img_size: (width, height) for adaptive threshold
        
    Returns:
        Boolean mask [T] indicating valid (not sparse) frames
    """
    T = len(kp_list)
    if T == 0:
        return np.array([], dtype=bool)
    
    valid_frames = np.ones(T, dtype=bool)
    
    for t, kp in enumerate(kp_list):
        if kp.shape[0] == 0:
            valid_frames[t] = False
            continue
            
        xy = kp[:, :2]
        conf = kp[:, 2]
        
        # Only consider confident keypoints
        confident_mask = conf > 0
        if not confident_mask.any():
            valid_frames[t] = False
            continue
            
        confident_keypoints = xy[confident_mask]
        
        # Calculate centroid
        centroid = np.mean(confident_keypoints, axis=0)
        
        # Calculate average distance from each keypoint to the centroid
        distances_to_centroid = np.sqrt(np.sum((confident_keypoints - centroid) ** 2, axis=1))
        avg_distance_to_centroid = np.mean(distances_to_centroid)
        
        # Adaptive threshold based on image size
        adaptive_threshold = min(img_size) * 0.1  # 10% of smaller image dimension
        threshold = max(sparse_threshold, adaptive_threshold)
        
        if avg_distance_to_centroid > threshold:
            frame_name = img_names[t] if t < len(img_names) else f'frame_{t:06d}'
            print(f"Discarding frame {frame_name} due to sparse keypoints (avg dist: {avg_distance_to_centroid:.2f} > {threshold:.2f})")
            valid_frames[t] = False
            
    return valid_frames

def process_2d_sequence(kp_list: list, speed_thresh: float, max_gap: int, max_edge_extrap: int, 
                       ma_window: int, huber_delta: float, sparse_threshold: float = 50.0, 
                       img_names: list = None, img_size=(640, 480)):
    """Process 2D hand keypoint sequence with smoothing and interpolation.
    
    Args:
        kp_list: list of [J, 3] arrays (x, y, conf) length T
        speed_thresh: max allowed center movement per frame in pixels
        max_gap: max gap length to interpolate
        max_edge_extrap: max extrapolation at sequence edges
        ma_window: moving average window size
        huber_delta: robust clipping threshold
        sparse_threshold: threshold for sparse keypoint detection
        img_names: list of image names for logging
        img_size: (width, height) for adaptive thresholds
        
    Returns:
        Tuple of (processed_keypoints_list, frame_valid_list)
    """
    T = len(kp_list)
    if T == 0:
        return kp_list, [False] * 0
    
    J = kp_list[0].shape[0]  # Should be 21 for hand keypoints
    kp = np.stack(kp_list)  # [T, J, 3]
    xy = kp[..., :2].astype(np.float32)
    conf = kp[..., 2]
    valid = conf > 0

    # Filter out frames with sparse keypoints
    #sparse_valid = filter_sparse_keypoints_2d(kp_list, img_names or [], sparse_threshold, img_size)
    
    # Compute hand centers and detect speed spikes
    #centers = compute_center_2d(xy, valid)
    #spikes = detect_speed_spikes_2d(centers, speed_thresh, max_interval=max_gap)

    # Mark spike frames and sparse frames as invalid for interpolation
    #valid[spikes] = False
    #valid[~sparse_valid] = False

    # Store original valid poses
    original_valid = valid.copy()
    
    # Interpolate each joint and dimension
    xy_filled = xy.copy().astype(np.float32)  # Start with original data
    interp_mask = np.zeros((T, J), dtype=bool)
    
    for j in range(J):
        for d in range(2):  # x, y coordinates
            series = xy[:, j, d]
            mask_j = valid[:, j]
            
            # Interpolate missing values
            interp_1d = interpolate_1d(series, mask_j, max_gap=max_gap, max_edge_extrap=max_edge_extrap)
            
            # Apply robust smoothing only to interpolated parts
            smooth_1d = moving_average_robust(interp_1d, window=ma_window, delta=huber_delta)
            
            # Keep original values for valid poses, use smoothed values for interpolated poses
            for t in range(T):
                if not mask_j[t]:  # Invalid original pose
                    if not np.isnan(smooth_1d[t]):  # Successfully interpolated
                        xy_filled[t, j, d] = smooth_1d[t]
                        interp_mask[t, j] = True
                    else:  # Could not interpolate
                        xy_filled[t, j, d] = 0.0
                        interp_mask[t, j] = False
                # else: keep original valid pose unchanged

    # Set confidence values: 1.0 for original valid, 0.5 for interpolated, 0 otherwise
    conf_out = np.zeros_like(conf, dtype=np.float32)
    conf_out[original_valid] = 1.0  # Original valid poses keep full confidence
    newly_filled = np.logical_and(~original_valid, interp_mask)
    conf_out[newly_filled] = 0.5  # Interpolated poses get medium confidence

    # Compose output list
    out_list = []
    frame_valid = []
    for t in range(T):
        xy_t = xy_filled[t]
        conf_t = conf_out[t]
        
        # Handle remaining NaNs (long gaps)
        nan_j = np.isnan(xy_t).any(axis=1)
        if nan_j.any():
            xy_t[nan_j] = 0.0
            conf_t[nan_j] = 0.0
            
        # Combine coordinates and confidence
        out = np.concatenate([xy_t, conf_t[..., None]], axis=1)
        out_list.append(out)
        
        # Frame is valid if at least half the joints are confident
        frame_valid.append((conf_t > 0).sum() >= (J // 2))
        
    return out_list, frame_valid

def load_hand_poses(handpose_file):
    """Load hand poses from pickle file and convert to standardized format.
    
    Args:
        handpose_file: path to handpose pickle file
        
    Returns:
        Tuple of (left_kp_list, right_kp_list, img_names)
    """
    with open(handpose_file, 'rb') as f:
        data = pickle.load(f)
    
    # Sort by image names to ensure temporal order
    img_names = sorted(data.keys())
    left_kp_list = []
    right_kp_list = []
    
    for img_name in img_names:
        hand_info = data[img_name]
        
        # Initialize empty keypoints
        left_kp = np.zeros((21, 3), dtype=np.float32)  # x, y, conf
        right_kp = np.zeros((21, 3), dtype=np.float32)
        
        # Fill in detected keypoints
        for hand_data in hand_info:
            if 'handness' in hand_data and 'keypoints' in hand_data:
                keypoints = hand_data['keypoints']  # [21, 2]
                handness = hand_data['handness']
                
                if handness == 'Left':
                    left_kp[:, :2] = keypoints
                    left_kp[:, 2] = 1.0  # Set confidence to 1.0 for detected keypoints
                elif handness == 'Right':
                    right_kp[:, :2] = keypoints
                    right_kp[:, 2] = 1.0
        
        left_kp_list.append(left_kp)
        right_kp_list.append(right_kp)
    
    return left_kp_list, right_kp_list, img_names

def main():
    parser = argparse.ArgumentParser(description='Smooth and interpolate 2D egocentric hand poses')
    parser.add_argument('--base_dir', type=str, required=True, help='dataset root containing the Data and Analysis folders')
    parser.add_argument('--groups', type=str, nargs='+', default=['Students', 'Residents', 'Attendings'])
    parser.add_argument('--subj_name', type=str, default=None, help='if set, only process this subject')
    parser.add_argument('--camera', type=str, default='Ego', help='camera name to process')
    parser.add_argument('--speed_thresh', type=float, default=100.0, help='max allowed center displacement per frame in pixels')
    parser.add_argument('--max_gap', type=int, default=8, help='max gap length to interpolate')
    parser.add_argument('--max_edge_extrap', type=int, default=5, help='max length to extrapolate at sequence edges')
    parser.add_argument('--ma_window', type=int, default=3, help='moving average window (should be odd)')
    parser.add_argument('--huber_delta', type=float, default=20.0, help='robust clipping for moving average in pixels')
    parser.add_argument('--sparse_threshold', type=float, default=500.0, help='threshold for sparse keypoint detection in pixels')
    parser.add_argument('--img_width', type=int, default=1920, help='image width for adaptive thresholds')
    parser.add_argument('--img_height', type=int, default=1080, help='image height for adaptive thresholds')
    parser.add_argument('--visualize', action='store_true', help='create visualization plots')
    parser.add_argument('--compare', action='store_true', help='create comparison visualization (original vs smoothed)')
    parser.add_argument('--out_suffix', type=str, default='smoothed', help='suffix for output files')
    
    args = parser.parse_args()

    data_dir = os.path.join(args.base_dir, 'Data')
    img_size = (args.img_width, args.img_height)

    for group in args.groups:
        group_dir = os.path.join(data_dir, group)
        if not os.path.isdir(group_dir):
            print(f"Group directory not found: {group_dir}")
            continue
            
        subj_names = sorted([f for f in os.listdir(group_dir) 
                           if os.path.isdir(os.path.join(group_dir, f)) and f[0].isdigit()])
        
        for subj in tqdm(subj_names, desc=f'Processing {group}'):
            if args.subj_name and subj != args.subj_name:
                continue
                
            base_dir = os.path.join(group_dir, subj)
            handpose_dir = os.path.join(base_dir, 'handpose')
            
            # Look for camera-specific handpose file
            handpose_file = os.path.join(handpose_dir, f'handpose_{args.camera}.pkl')
            if not os.path.exists(handpose_file):
                print(f"Handpose file not found: {handpose_file}")
                continue
            
            print(f"Processing {group}/{subj} - {args.camera}")
            
            # Load hand poses
            left_kp_list, right_kp_list, img_names = load_hand_poses(handpose_file)
            
            if len(left_kp_list) == 0:
                print(f"No keypoints found in {handpose_file}")
                continue
            
            # Store original data for comparison
            original_data = {
                'left': left_kp_list,
                'right': right_kp_list,
                'image': img_names
            }
            
            # Process left and right hands
            left_proc, left_valid = process_2d_sequence(
                left_kp_list, args.speed_thresh, args.max_gap, args.max_edge_extrap,
                args.ma_window, args.huber_delta, args.sparse_threshold, img_names, img_size
            )
            
            right_proc, right_valid = process_2d_sequence(
                right_kp_list, args.speed_thresh, args.max_gap, args.max_edge_extrap,
                args.ma_window, args.huber_delta, args.sparse_threshold, img_names, img_size
            )
            
            # Save processed data
            out_data = {
                'left': left_proc,
                'right': right_proc,
                'image': img_names,
                'left_valid': left_valid,
                'right_valid': right_valid,
            }
            
            out_path = os.path.join(handpose_dir, f'handpose_{args.camera}_{args.out_suffix}.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(out_data, f)
            
            print(f"Saved smoothed poses to: {out_path}")
            
            # Visualization if requested
            if (args.visualize or args.compare) and (args.subj_name is None or subj == args.subj_name):
                if args.visualize:
                    vis_dir = os.path.join(handpose_dir, f'vis2d_{args.camera}_{args.out_suffix}')
                    visualize_2d_sequence(out_data, vis_dir, base_dir, args.camera, img_size)
                    print(f"Saved visualizations to: {vis_dir}")
                
                if args.compare:
                    comp_dir = os.path.join(handpose_dir, f'vis2d_{args.camera}_comparison')
                    visualize_2d_comparison(out_data, original_data, comp_dir, base_dir, args.camera, img_size)
                    print(f"Saved comparison visualizations to: {comp_dir}")

if __name__ == '__main__':
    main()
