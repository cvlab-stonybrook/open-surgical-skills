import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

# Example usage: python triangulation/smooth_hand3d.py --base_dir $DATA_ROOT --groups Attendings --subj_name 09_30 --cam_as_world Cam3 --use_easymocap_vis (omit the last argument if not visualizing)


# Robust utilities

# Visualization utilities
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import cv2
import sys

# Ensure local EasyMocap package is on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_EASyMOCAP_ROOT = os.path.join(_REPO_ROOT, 'third_party', 'EasyMocap')
for p in [_REPO_ROOT, _EASyMOCAP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional EasyMocap imports for reprojection-based vis
try:
    from easymocap.mytools import projectN3
    from easymocap.dataset import CONFIG, MV1P2H
    EASYMOCAP_AVAILABLE = True
except Exception as e:
    print('EasyMocap import failed:', repr(e))
    EASYMOCAP_AVAILABLE = False

# Hand skeleton edges for 21-joint topology (wrist=0)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def set_axes_equal_3d(ax, X):
    # X: [N,3]
    if X.size == 0:
        return
    x_limits = [np.nanmin(X[:, 0]), np.nanmax(X[:, 0])]
    y_limits = [np.nanmin(X[:, 1]), np.nanmax(X[:, 1])]
    z_limits = [np.nanmin(X[:, 2]), np.nanmax(X[:, 2])]
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range, 1e-6)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)


def plot_hand(ax, pts, color):
    # pts: [21,4], last column is confidence
    conf = pts[:, 3] > 0
    xyz = pts[:, :3]
    # joints
    ax.scatter(xyz[conf, 0], xyz[conf, 1], xyz[conf, 2], c=color, s=8, depthshade=False)
    # edges
    for i, j in HAND_EDGES:
        if conf[i] and conf[j]:
            xs = [xyz[i, 0], xyz[j, 0]]
            ys = [xyz[i, 1], xyz[j, 1]]
            zs = [xyz[i, 2], xyz[j, 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=1)


def visualize_sequence(out_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    left = out_data['left']
    right = out_data['right']
    names = out_data.get('image', [])
    T = min(len(left), len(right))
    for t in tqdm(range(T), desc='Render smoothed 3D hand plots'):
        L = left[t]
        R = right[t]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        plot_hand(ax, L, color='tab:blue')
        plot_hand(ax, R, color='tab:red')
        # Equal axes based on confident joints from both hands
        both = []
        for P in (L, R):
            mask = P[:, 3] > 0
            both.append(P[mask, :3])
        both = np.concatenate(both, axis=0) if len(both) and all(b.size for b in both) else np.empty((0, 3))
        set_axes_equal_3d(ax, both)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=60)
        title = names[t] if t < len(names) else f'{t:06d}.jpg'
        ax.set_title(f'Smoothed 3D Hands - {title}')
        out_name = os.path.splitext(title)[0] + '.png'
        out_path = os.path.join(out_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)


def visualize_with_repro_easymocap(subj_path, out_data, image_root='Images', annot_root='handpose', annot_file='handpose.pkl', cams=None, out_dir=None, cam_as_world='Cam3'):
    if not EASYMOCAP_AVAILABLE:
        raise RuntimeError('EasyMocap is not available in this environment.')
    cams = cams or sorted([d for d in os.listdir(os.path.join(subj_path, image_root)) if d.startswith('Cam') and len(d) == 4])
    out_dir = out_dir or os.path.join(subj_path, "handpose", 'vis3d_smoothed')
    os.makedirs(out_dir, exist_ok=True)
    cam_vis = ['Cam2', 'Cam3', 'Cam4']
    subj_name = os.path.basename(subj_path)
    dataset = MV1P2H(subj_path, image_root=image_root, annot_root=annot_root, annot_file=annot_file,
        cams=cams, out=out_dir, config=CONFIG['handl'], kpts_type='handl', undis=False, no_img=False,
        start_frame=0, end_frame=-1, verbose=False, cam_as_world=cam_as_world)

    left = out_data['left']
    right = out_data['right']
    names = out_data.get('image', [])
    T = min(len(left), len(right))

    for idx in tqdm(range(T), desc='Reproj vis (EasyMocap)'):
        try:
            images, img_idx, annots_left, annots_right, valid_left, valid_right = dataset[idx]
        except Exception:
            # Fallback shape without annots
            print('Fallback shape without annots')
            images, img_idx = dataset[idx][:2]
            valid_left = None
            valid_right = None
        L = left[idx][:, :3]
        R = right[idx][:, :3]
        
        valid_left = np.ones_like(valid_left).astype(bool)
        valid_right = np.ones_like(valid_right).astype(bool)
        valid_left[0] = False
        valid_right[0] = False
        
        P_left = dataset.Pall[valid_left, ...]
        P_right = dataset.Pall[valid_right, ...]
    
        kpts_repro_l = projectN3(L, P_left) 
        kpts_repro_r = projectN3(R, P_right)
        
        conf_left = left[idx][None, :, -1:].repeat(len(P_left), axis=0)
        kpts_repro_l = np.concatenate((kpts_repro_l, conf_left), axis=2)
        kpts_repro_l[kpts_repro_l[:,:,-1]==0] = 0.0
        
        conf_right = right[idx][None, :, -1:].repeat(len(P_right), axis=0)
        kpts_repro_r = np.concatenate((kpts_repro_r, conf_right), axis=2) # [N, 21, 4]
        kpts_repro_r[kpts_repro_r[:,:,-1]==0] = 0.0

        # Draw on fused canvas
        
        img = dataset.vis_repro(images, kpts_repro_l, nf=img_idx, sub_vis=cam_vis, onehand=True, valid_note=valid_left)
        img = dataset.vis_repro(img, kpts_repro_r, nf=img_idx, sub_vis=cam_vis, onehand=False, kp_3d=[left[idx], right[idx]], valid_note=valid_right)
        # Image already saved by EasyMocap writer via vis_repro to <out>/repro/{nf:06d}.jpg


def compute_center(poses_xyz: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    # poses_xyz: [T, J, 3]; valid_mask: [T, J]
    T, J, _ = poses_xyz.shape
    centers = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        mask_t = valid_mask[t]
        if mask_t.any():
            centers[t] = poses_xyz[t, mask_t].mean(axis=0)
        else:
            centers[t] = 0.0
    return centers


def detect_speed_spikes(centers: np.ndarray, speed_thresh: float, max_interval: int = None) -> np.ndarray:
    # centers: [T,3]
    # Compare each frame with the previous valid frame instead of just neighboring frames
    # This handles cases where multiple consecutive frames are incorrectly predicted
    # max_interval: maximum allowed interval between current frame and last valid frame
    # If exceeded, fall back to comparing with immediate previous frame
    T = centers.shape[0]
    spikes = np.zeros(T, dtype=bool)
    
    if T <= 1:
        return spikes
    
    last_valid_idx = 0  # Start with the first frame as valid
    
    for i in range(1, T):
        reference_idx = last_valid_idx
        
        # If max_interval is specified and exceeded, use immediate previous frame instead
        if max_interval is not None and (i - last_valid_idx) > max_interval:
            reference_idx = i - 1
        
        # Compute distance from current frame to reference frame
        delta = np.linalg.norm(centers[i] - centers[reference_idx])
        
        if delta > speed_thresh:
            # Mark current frame as a spike
            spikes[i] = True
        else:
            # Current frame is valid, update last valid frame index
            last_valid_idx = i
    
    return spikes


def find_invalid_runs(mask: np.ndarray) -> list:
    # mask False means invalid; returns list of (start, end) inclusive indices for invalid runs
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
    # values: [T], may contain arbitrary numbers; valid indicates which entries are trustworthy
    T = values.shape[0]
    if valid.sum() == 0:
        return np.full(T, np.nan, dtype=np.float32)
    
    # Start with original values
    out = values.copy().astype(np.float32)
    
    # Only interpolate invalid indices
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
            # leading gap: allow up to max_edge_extrap
            if run_len > max_edge_extrap:
                out[: e - max_edge_extrap + 1] = np.nan
        elif e == T - 1:
            if run_len > max_edge_extrap:
                out[s + max_edge_extrap :] = np.nan
        if run_len > max_gap and s != 0 and e != T - 1:
            out[s : e + 1] = np.nan
    return out


def find_valid_segments(valid: np.ndarray, min_segment_len: int = 10) -> list:
    """Find continuous segments of valid data that are long enough to be useful for interpolation.
    
    Args:
        valid: [T] boolean array indicating valid frames
        min_segment_len: minimum length for a segment to be considered valid
        
    Returns:
        List of (start, end) tuples for valid segments (inclusive indices)
    """
    segments = []
    T = valid.shape[0]
    i = 0
    while i < T:
        if valid[i]:
            j = i
            while j + 1 < T and valid[j + 1]:
                j += 1
            if j - i + 1 >= min_segment_len:  # Only keep segments long enough
                segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


def interpolate_1d_segmented(values: np.ndarray, valid: np.ndarray, max_gap: int, max_edge_extrap: int, 
                           max_segment_gap: int = 30, min_segment_len: int = 10) -> np.ndarray:
    """Interpolate 1D values using segment-based approach for better temporal locality.
    
    Args:
        values: [T] array of values to interpolate
        valid: [T] boolean array indicating trustworthy entries  
        max_gap: max gap length to interpolate within segments
        max_edge_extrap: max length to extrapolate at segment edges
        max_segment_gap: max gap between segments to bridge
        min_segment_len: minimum segment length to be useful for interpolation
        
    Returns:
        [T] interpolated array with NaN for unfillable gaps
    """
    T = values.shape[0]
    if valid.sum() == 0:
        return np.full(T, np.nan, dtype=np.float32)
    
    # Find valid segments
    segments = find_valid_segments(valid, min_segment_len)
    if not segments:
        return np.full(T, np.nan, dtype=np.float32)
    
    out = np.full(T, np.nan, dtype=np.float32)
    
    # Process each segment independently
    for seg_start, seg_end in segments:
        # Extract segment data
        seg_len = seg_end - seg_start + 1
        seg_values = values[seg_start:seg_end + 1]
        seg_valid = valid[seg_start:seg_end + 1]
        
        # Interpolate within this segment using the original logic
        seg_out = interpolate_1d(seg_values, seg_valid, max_gap, max_edge_extrap)
        out[seg_start:seg_end + 1] = seg_out
    
    # Bridge small gaps between segments if they're close enough
    if len(segments) > 1:
        for i in range(len(segments) - 1):
            _, end1 = segments[i]
            start2, _ = segments[i + 1]
            gap_len = start2 - end1 - 1
            
            if gap_len <= max_segment_gap and gap_len > 0:
                # Bridge the gap with linear interpolation
                gap_start = end1 + 1
                gap_end = start2 - 1
                
                if not np.isnan(out[end1]) and not np.isnan(out[start2]):
                    # Linear interpolation between segments
                    gap_indices = np.arange(gap_start, gap_end + 1)
                    interp_values = np.interp(gap_indices, [end1, start2], [out[end1], out[start2]])
                    out[gap_start:gap_end + 1] = interp_values
    
    return out


def moving_average_robust(values: np.ndarray, window: int, delta: float) -> np.ndarray:
    # values: [T] with NaNs
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


def filter_sparse_keypoints(kp_list: list, img_names: list, sparse_threshold_factor: float = 0.08) -> np.ndarray:
    """Filter out frames with overly sparse keypoints (too spread out).
    
    Args:
        kp_list: list of [J,4] arrays length T
        sparse_threshold_factor: threshold as fraction of pose scale for sparse detection
        
    Returns:
        Boolean mask [T] indicating which frames are valid (not sparse)
    """
    T = len(kp_list)
    if T == 0:
        return np.array([], dtype=bool)
    
    valid_frames = np.ones(T, dtype=bool)
    
    for t, kp in enumerate(kp_list):
        if kp.shape[0] == 0:
            valid_frames[t] = False
            continue
            
        xyz = kp[:, :3]
        conf = kp[:, 3]
        
        # Only consider confident keypoints
        confident_mask = conf > 0
        if not confident_mask.any():
            valid_frames[t] = False
            continue
            
        confident_keypoints = xyz[confident_mask]
        
        # Calculate centroid (average point) of the hand pose
        centroid = np.mean(confident_keypoints, axis=0)
        
        # Calculate average distance from each keypoint to the centroid
        distances_to_centroid = np.sqrt(np.sum((confident_keypoints - centroid) ** 2, axis=1))
        avg_distance_to_centroid = np.mean(distances_to_centroid)
        
        # Calculate pose scale for adaptive threshold
        #pose_range = np.ptp(confident_keypoints, axis=0)  # range per dimension
        #pose_scale = np.max(pose_range)  # use max range as scale
        
        # Threshold for sparse keypoints (adaptive based on pose scale)
        #sparse_threshold = pose_scale * sparse_threshold_factor
        sparse_threshold = 8
        if avg_distance_to_centroid > sparse_threshold:
            frame_name = img_names[t]
            print(f"Discarding frame {frame_name} due to sparse keypoints (avg dist to centroid: {avg_distance_to_centroid:.2f} > {sparse_threshold:.2f})")
            valid_frames[t] = False
            
    return valid_frames


def process_sequence(kp_list: list, speed_thresh: float, max_gap: int, max_edge_extrap: int, ma_window: int, huber_delta: float, 
                    max_segment_gap: int = 30, min_segment_len: int = 10, use_global_interp: bool = False, 
                    sparse_threshold_factor: float = 0.08, img_names: list = None):
    # kp_list: list of [J,4] arrays length T
    T = len(kp_list)
    if T == 0:
        return kp_list, [False] * 0
    J = kp_list[0].shape[0]
    kp = np.stack(kp_list)  # [T,J,4]
    xyz = kp[..., :3].astype(np.float32)
    conf = kp[..., 3]
    valid = conf > 0

    # Filter out frames with sparse keypoints
    sparse_valid = filter_sparse_keypoints(kp_list, img_names, sparse_threshold_factor)
    
    centers = compute_center(xyz, valid)
    spikes = detect_speed_spikes(centers, speed_thresh, max_interval=max_gap)

    # Mark spike frames and sparse frames as invalid for interpolation (conservative)
    valid[spikes] = False
    valid[~sparse_valid] = False

    # Interpolate each joint and dim
    xyz_filled = np.full_like(xyz, np.nan, dtype=np.float32)
    interp_mask = np.zeros((T, J), dtype=bool)
    for j in range(J):
        for d in range(3):
            series = xyz[:, j, d]
            mask_j = valid[:, j]
            if use_global_interp:
                interp_1d = interpolate_1d(series, mask_j, max_gap=max_gap, max_edge_extrap=max_edge_extrap)
            else:
                interp_1d = interpolate_1d_segmented(series, mask_j, max_gap, max_edge_extrap, max_segment_gap, min_segment_len)
            smooth_1d = moving_average_robust(interp_1d, window=ma_window, delta=huber_delta)
            xyz_filled[:, j, d] = smooth_1d
        interp_mask[:, j] = ~np.isnan(xyz_filled[:, j, 0])

    # Confidence: 1.0 original valid, 0.5 interpolated, 0 otherwise
    conf_out = np.zeros_like(conf, dtype=np.float32)
    conf_out[valid] = 1.0
    newly_filled = np.logical_and(~valid, interp_mask)
    conf_out[newly_filled] = 0.5

    # Compose output list
    out_list = []
    frame_valid = []
    for t in range(T):
        xyz_t = xyz_filled[t]
        conf_t = conf_out[t]
        # If still NaNs exist (long gaps), set them to 0 and conf 0
        nan_j = np.isnan(xyz_t).any(axis=1)
        if nan_j.any():
            xyz_t[nan_j] = 0.0
            conf_t[nan_j] = 0.0
        out = np.concatenate([xyz_t, conf_t[..., None]], axis=1)
        out_list.append(out)
        # A frame is considered valid if at least half joints are nonzero after processing
        frame_valid.append((conf_t > 0).sum() >= (J // 2))
    return out_list, frame_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='dataset root containing the Data and Analysis folders')
    parser.add_argument('--groups', type=str, nargs='+', default=['Students', 'Residents', 'Attendings'])
    parser.add_argument('--subj_name', type=str, default=None, help='if set, only process and visualize this subject')
    parser.add_argument('--use_easymocap_vis', action='store_true', help='use EasyMocap vis_repro to render reprojections per view (requires --subj_name)')
    parser.add_argument('--imagefolder', type=str, default='Images')
    parser.add_argument('--annotfolder', type=str, default='handpose')
    parser.add_argument('--annot_file', type=str, default='handpose_track.pkl')
    parser.add_argument('--speed_thresh', type=float, default=100.0, help='max allowed center displacement per frame in world units')
    parser.add_argument('--max_gap', type=int, default=8, help='max gap length to interpolate inside the sequence')
    parser.add_argument('--max_edge_extrap', type=int, default=5, help='max length to extrapolate at the sequence edges')
    parser.add_argument('--ma_window', type=int, default=3, help='moving average window (odd)')
    parser.add_argument('--huber_delta', type=float, default=100.0, help='robust clipping for moving average')
    parser.add_argument('--cam_as_world', type=str, default='Cam3', help='the camera to use as the world coordinate system')
    parser.add_argument('--max_segment_gap', type=int, default=30, help='max gap between segments to bridge with interpolation')
    parser.add_argument('--min_segment_len', type=int, default=5, help='minimum segment length to be useful for interpolation')
    parser.add_argument('--use_global_interp', action='store_true', help='use original global interpolation instead of segment-based')
    parser.add_argument('--sparse_threshold_factor', type=float, default=0.1, help='threshold factor for detecting sparse keypoints (fraction of pose scale)')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory for visualization')
    args = parser.parse_args()

    data_dir = os.path.join(args.base_dir, 'Data')

    for group in args.groups:
        group_dir = os.path.join(data_dir, group)
        if not os.path.isdir(group_dir):
            continue
        subj_names = sorted([f for f in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, f))])
        for subj in tqdm(subj_names, desc=f'Postprocess {group}'):
            if args.subj_name and subj != args.subj_name:
                continue
            base_dir = os.path.join(group_dir, subj)
            in_path = os.path.join(base_dir, 'handpose', 'keypoints_3d.pickle')
            if not os.path.exists(in_path):
                continue
            with open(in_path, 'rb') as f:
                kp_3d = pickle.load(f)
            
            left_list = kp_3d['left']
            right_list = kp_3d['right']
            img_names = kp_3d['image']
            assert len(left_list) == len(right_list), f"Left and right hand lists have different lengths: {len(left_list)} != {len(right_list)}"
            num_frames = len(os.listdir(os.path.join(base_dir, args.imagefolder, "Cam2")))
            if len(left_list) != num_frames:
                print(f"Warning: Left and right hand lists have different lengths: {len(left_list)} != {num_frames}")
                start_frame = len(left_list) - num_frames
                left_list = left_list[start_frame:]
                right_list = right_list[start_frame:]

            left_proc, left_valid = process_sequence(left_list, args.speed_thresh, args.max_gap, args.max_edge_extrap, args.ma_window, args.huber_delta,
                                                    args.max_segment_gap, args.min_segment_len, args.use_global_interp, args.sparse_threshold_factor, img_names=img_names)
            right_proc, right_valid = process_sequence(right_list, args.speed_thresh, args.max_gap, args.max_edge_extrap, args.ma_window, args.huber_delta,
                                                     args.max_segment_gap, args.min_segment_len, args.use_global_interp, args.sparse_threshold_factor, img_names=img_names)
            
            out_data = {
                'left': left_proc,
                'right': right_proc,
                'image': kp_3d.get('image', []),
                'left_valid': left_valid,
                'right_valid': right_valid,
            }
            out_path = os.path.join(base_dir, 'handpose', 'keypoints_3d_smoothed.pickle')
            with open(out_path, 'wb') as f:
                pickle.dump(out_data, f)

            # Visualization for specified subject
            if args.subj_name and subj == args.subj_name:
                if args.use_easymocap_vis:
                    if not EASYMOCAP_AVAILABLE:
                        print('EasyMocap not available, skipping vis_repro-based visualization.')
                    else:
                        out_dir = os.path.join(base_dir, "handpose", 'vis3d_smoothed') if args.out_dir is None else os.path.join(base_dir, 'handpose', args.out_dir)
                        visualize_with_repro_easymocap(base_dir, out_data, image_root=args.imagefolder, annot_root=args.annotfolder, annot_file=args.annot_file, cams=None, out_dir=out_dir)
                        
                else:
                    vis_dir = os.path.join(base_dir, 'handpose_smoothed')
                    visualize_sequence(out_data, vis_dir)

if __name__ == '__main__':
    main() 