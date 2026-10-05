import os
import argparse
import cv2
import glob
import pickle
import torch
import numpy as np
from tqdm import tqdm

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# Add matplotlib for simple visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def validate_and_fix_tracked_keypoints(pred_kp, pred_vis, hand_name, img_name, prev_keypoints=None, threshold=20):
    """
    Validate tracked keypoints and fix outliers by clustering and temporal consistency.
    
    Args:
        pred_kp: predicted keypoints array (N, 2)
        pred_vis: visibility mask for keypoints (N,)
        hand_name: name of the hand (for logging)
        img_name: image name (for logging)
        prev_keypoints: keypoints from previous frame for temporal interpolation (N, 2)
        threshold: distance threshold for sparse detection
    
    Returns:
        tuple: (corrected_keypoints, is_valid, message)
        - corrected_keypoints: fixed keypoints array
        - is_valid: boolean indicating if hand should be kept
        - message: logging message
    """
    # Filter out invalid keypoints (negative coordinates)
    valid_keypoints = pred_kp[pred_vis]
    
    if len(valid_keypoints) == 0:
        return pred_kp, False, f"no valid keypoints"
    
    # Check 1: Average distance to centroid
    centroid = np.mean(valid_keypoints, axis=0)
    distances_to_centroid = np.sqrt(np.sum((valid_keypoints - centroid) ** 2, axis=1))
    avg_distance_to_centroid = np.mean(distances_to_centroid)
    sparse_threshold = threshold 
    
    if avg_distance_to_centroid <= sparse_threshold:
        # Keypoints are fine, use as is
        return pred_kp, True, f"keypoints are valid (avg dist: {avg_distance_to_centroid:.2f})"
    
    # Keypoints are sparse, try to fix them
    print(f"Fixing sparse tracked keypoints for {hand_name} hand in {img_name} (avg dist: {avg_distance_to_centroid:.2f} > {sparse_threshold:.2f})")
    
    # Find the main cluster of keypoints (likely the actual hand)
    cluster_threshold = sparse_threshold * 0.8  # Points within this distance are considered clustered
    point_neighbors = []
    
    for i, pt in enumerate(valid_keypoints):
        neighbors = 0
        for j, other_pt in enumerate(valid_keypoints):
            if i != j:
                dist = np.sqrt(np.sum((pt - other_pt) ** 2))
                if dist < cluster_threshold:
                    neighbors += 1
        point_neighbors.append(neighbors)
    
    # Find points that have the most neighbors (core cluster)
    max_neighbors = max(point_neighbors) if point_neighbors else 0
    if max_neighbors < 3:  # Need at least 3 neighbors to form a cluster
        return pred_kp, False, f"no clear cluster found (max neighbors: {max_neighbors})"
    
    # Select points that are in the main cluster
    cluster_indices = []
    for i, neighbors in enumerate(point_neighbors):
        if neighbors >= max_neighbors * 0.5:  # Points with 50% of max neighbors
            cluster_indices.append(i)
    
    if len(cluster_indices) < 5:  # Need at least 5 points for a valid hand cluster
        return pred_kp, False, f"insufficient cluster size ({len(cluster_indices)})"
    
    cluster_points = valid_keypoints[cluster_indices]
    cluster_centroid = np.mean(cluster_points, axis=0)
    
    # Create corrected keypoints
    corrected_pred_kp = pred_kp.copy()
    valid_indices = np.where(pred_vis)[0]
    outlier_count = 0
    
    # For outlier points, use temporal information if available
    if prev_keypoints is not None and len(prev_keypoints) == len(pred_kp):
        # Calculate transformation from previous frame to current inlier cluster
        prev_valid_cluster = prev_keypoints[valid_indices[cluster_indices]]
        
        prev_cluster_centroid = np.mean(prev_valid_cluster, axis=0)
        
        # Calculate translation and scale from previous to current cluster
        translation = cluster_centroid - prev_cluster_centroid
        
        # Calculate scale factor based on cluster size change
        prev_cluster_distances = np.sqrt(np.sum((prev_valid_cluster - prev_cluster_centroid) ** 2, axis=1))
        curr_cluster_distances = np.sqrt(np.sum((cluster_points - cluster_centroid) ** 2, axis=1))
        
        if np.mean(prev_cluster_distances) > 0:
            scale_factor = np.mean(curr_cluster_distances) / np.mean(prev_cluster_distances)
        else:
            scale_factor = 1.0
        
        # Apply transformation to outlier keypoints
        for i, (valid_idx, point) in enumerate(zip(valid_indices, valid_keypoints)):
            if i not in cluster_indices:  # This is an outlier
                prev_point = prev_keypoints[valid_idx]
                
                # Transform the previous keypoint using the same transformation as the cluster
                # 1. Translate relative to previous cluster centroid
                relative_pos = prev_point - prev_cluster_centroid
                # 2. Scale the relative position
                scaled_pos = relative_pos * scale_factor
                # 3. Translate to current cluster centroid
                estimated_pos = cluster_centroid + scaled_pos
                
                corrected_pred_kp[valid_idx] = estimated_pos
                outlier_count += 1
    else:
        # Fallback: simple interpolation towards cluster centroid
        for i, (valid_idx, point) in enumerate(zip(valid_indices, valid_keypoints)):
            if i not in cluster_indices:  # This is an outlier
                # Simple interpolation: move outlier towards cluster centroid
                direction = point - centroid
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    # Place outlier at cluster edge
                    new_pos = cluster_centroid + direction * (sparse_threshold * 0.6)
                    corrected_pred_kp[valid_idx] = new_pos
                else:
                    corrected_pred_kp[valid_idx] = cluster_centroid
                outlier_count += 1
    
    method = "temporal interpolation" if prev_keypoints is not None else "cluster-based interpolation"
    return corrected_pred_kp, True, f"fixed {outlier_count} outlier keypoints using {method}"

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, required=True, help='dataset root containing the Data and Analysis folders')
parser.add_argument('--group', type=str, default='Students', help='students or residents or doctors')
parser.add_argument("--subj_folder", type=str, required=True, help='folder for subject containing images for different cameras')
#parser.add_argument('--cam', required=True)
parser.add_argument('--ckpt_path', type=str, required=True, help='path to the CoTracker checkpoint (cotracker_stride_4_wind_8.pth)')
parser.add_argument("--camera", type=str, default='')
parser.add_argument('--fps', type=int, default= 5, help='the fps for loading the video for processing')
parser.add_argument('--fps_ori', type=int, default= 30, help='original fps in video recordings')
parser.add_argument('--vis', action='store_true', help='enable visualization of tracked hands')
args = parser.parse_args()

subj_dir = os.path.join(args.base_dir, 'Data', args.group, args.subj_folder, "Images")
result_dir = os.path.join(args.base_dir, 'Data', args.group, args.subj_folder, 'handpose')

# Define hand landmark connections (simplified version of MediaPipe's hand connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm connections
]

if args.camera:
    all_cams = [args.camera]
else:
    all_cams = [each for each in os.listdir(subj_dir) if each.startswith('Cam')]

for cam_name in all_cams:
    img_dir = os.path.join(subj_dir, cam_name)
    output_dir = os.path.join(result_dir, cam_name+"_tracked")
    
    # Create output directory for visualization if needed
    if args.vis:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    with open(os.path.join(result_dir, f'handpose_{cam_name}.pkl'), 'rb') as file:
        handpose_dct = pickle.load(file)

    
    # replace .png with .jpg
    has_png = False
    handpose_dct_jpg = {}
    for imgname, info in handpose_dct.items():
        if '.png' in imgname:
            has_png = True
            imgname = imgname.replace('.png', '.jpg')
            handpose_dct_jpg[imgname] = info
    if has_png:
        handpose_dct = handpose_dct_jpg
    
    handness_info = {}
    for idx, (imgname, hand_info) in enumerate(handpose_dct.items()):
        handness_info[imgname] = {}
        for each_info in hand_info:
            handness_info[imgname][each_info['handness']] = each_info['keypoints']
        
    imgnames = sorted(list(handpose_dct.keys()))
    h,w = 720, 1280

    # do tracking for 5 consecutive frames
    seq_len = 5
    model = CoTrackerPredictor(
        checkpoint=args.ckpt_path
    )
    model = model.cuda()
    track_stored = {'Left':{}, 'Right':{}}

    for idx, imgname in enumerate(tqdm(imgnames)):
        hand_info = handpose_dct[imgname]
        
        if idx==0 or idx>=len(imgnames)-2:
            continue
        img_idx = int(os.path.splitext(imgname)[0])
        
        new_hand_info = []
        if cam_name.lower().startswith('cam1'):
            for each_info in hand_info:
                keypoints = each_info['keypoints']
                is_valid_hand = True
                discard_reason = ""
                
                # Check 1: Average distance to centroid
                centroid = np.mean(keypoints, axis=0)
                distances_to_centroid = np.sqrt(np.sum((keypoints - centroid) ** 2, axis=1))
                avg_distance_to_centroid = np.mean(distances_to_centroid)
                sparse_threshold = min(h, w) * 0.08  # 8% of the smaller image dimension
                
                if avg_distance_to_centroid > sparse_threshold:
                    is_valid_hand = False
                    discard_reason = f"sparse keypoints (avg dist to centroid: {avg_distance_to_centroid:.2f} > {sparse_threshold:.2f})"
                
                if is_valid_hand:
                    new_hand_info.append(each_info)
                else:
                    print(f"Discarding {each_info['handness']} hand for {imgname} due to {discard_reason}")
            handpose_dct[imgname] = new_hand_info
            hand_info = handpose_dct[imgname]
            
        missing_hands = ['Left', 'Right']
        # cases when detected both hands with one same side:
        if len(hand_info)==2:
            hands_unique = set([each['handness'] for each in hand_info])
            if len(hands_unique)==1:
                x1, x2 = np.mean(hand_info[0]['keypoints'][:,0]), np.mean(hand_info[1]['keypoints'][:,0])
                y1, y2 = np.mean(hand_info[0]['keypoints'][:,1]), np.mean(hand_info[1]['keypoints'][:,1])
                if cam_name=='Cam4':
                    hand_1st, hand_2nd = ('Left', 'Right') if y1 < y2 else ('Right', 'Left')
                    hand_info[0]['handness']= hand_1st
                    hand_info[1]['handness']= hand_2nd
                else:
                    hand_1st, hand_2nd = ('Left', 'Right') if x1 > x2 else ('Right', 'Left')
                    hand_info[0]['handness']= hand_1st
                    hand_info[1]['handness']= hand_2nd
                    
        if len(hand_info)>0:
            for each_info in hand_info:
                handness = each_info['handness']
                missing_hands.remove(handness)
        
        for hand_missing in missing_hands:

            
            if img_idx in track_stored[hand_missing]: 
                pred_kp, pred_vis = track_stored[hand_missing][img_idx]['pred_kp'], track_stored[hand_missing][img_idx]['pred_vis']
                num_kps = pred_vis.shape[0]
                if pred_vis.sum() < num_kps//2:
                    continue
                
                # Get previous frame keypoints for temporal interpolation
                # Search backwards through previous frames to find the same handness
                prev_kp = None
                for look_back in range(1, min(idx + 1, 10)):  # Look back up to 10 frames
                    prev_idx = idx - look_back
                    prev_key = imgnames[prev_idx]
                    for each_prev in handpose_dct[prev_key]:
                        if each_prev['handness'] == hand_missing:
                            prev_kp = each_prev['keypoints']
                            break
                    if prev_kp is not None:
                        break
                
                # Validate and potentially fix stored tracked keypoints
                if cam_name.lower().startswith('cam1'):
                    corrected_kp, is_valid, message = validate_and_fix_tracked_keypoints(
                        pred_kp, pred_vis, hand_missing, imgname, prev_keypoints=prev_kp, threshold=20
                    )
                else:
                    corrected_kp, is_valid, message = pred_kp, True, "valid"
                    
                if is_valid:
                    hand_append = {'handness': hand_missing, 'keypoints': corrected_kp, 'gesture': "Track"}
                    hand_info.append(hand_append)
                    #print(f"Using stored tracked {hand_missing} hand for {imgname}: {message}")
                else:
                    print(f"Discarding stored tracked {hand_missing} hand for {imgname} due to {message}")
                continue
            
            last_key = imgnames[idx-1]
            found = False
            for each_prev in handpose_dct[last_key]:
                if each_prev['handness'] == hand_missing:
                    found=True
                    last_pose = each_prev['keypoints']
            if not found:
                continue

            last_pose = last_pose[:,:2]
            last_pose = np.concatenate((np.zeros((last_pose.shape[0], 1)), last_pose), axis=1)  # first column in query corresponds to the starting frame, here we set 0 for all
            num_kps = last_pose.shape[0]
            if np.sum(last_pose[:,1]<0)>=num_kps//2 or np.sum(last_pose[:,2]<0)>=num_kps//2:
                continue
            
            img_seq, seq_indices = [],[]
            this_len = min(seq_len, len(imgnames)-idx+1)
            for i in range(this_len):
                img_this = imgnames[idx-1+i]
                imgpath = os.path.join(img_dir, img_this)
                seq_indices.append(int(os.path.splitext(img_this)[0]))
                img = cv2.imread(imgpath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_seq.append(img)   
            img_seq = np.stack(img_seq)
            img_seq = torch.from_numpy(img_seq).permute(0, 3, 1, 2)[None].float() 
            img_seq = img_seq.cuda()
                
            last_pose = torch.from_numpy(last_pose).float()
            num_kps = last_pose.shape[0]
            valid_idx = torch.logical_and(last_pose[:,1]>=0, last_pose[:,2]>=0) 
            last_pose = last_pose[valid_idx]
            last_pose = last_pose.cuda()
            pred_tracks, pred_visibility = model(img_seq, queries=last_pose[None])
            pred_visibility = pred_visibility[0].cpu()
            
            pred_tracks = pred_tracks[0].cpu()
            last_idx = int(os.path.splitext(last_key)[0])
            for i in range(this_len):
                pred_kp_all, pred_vis_all = torch.ones(num_kps, 2) * -1, torch.zeros(num_kps).bool()
                pred_kp_all[valid_idx] = pred_tracks[i]
                pred_vis_all[valid_idx] = pred_visibility[i]
                track_stored[hand_missing][seq_indices[i]] = {'pred_kp': pred_kp_all.numpy(), 'pred_vis': pred_vis_all.numpy()}
            
            # Validate and potentially fix newly tracked keypoints
            pred_kp = track_stored[hand_missing][img_idx]['pred_kp']
            pred_vis = track_stored[hand_missing][img_idx]['pred_vis']
            
            # Get previous frame keypoints for temporal interpolation
            prev_kp = last_pose[:, 1:3].cpu().numpy()  # Extract x,y coordinates from last_pose (removing frame index)
            
            # Only validate if we have enough visible keypoints
            if pred_vis.sum() >= len(pred_vis) // 2:
                if cam_name.lower().startswith('cam1'):
                    corrected_kp, is_valid, message = validate_and_fix_tracked_keypoints(
                        pred_kp, pred_vis, hand_missing, imgname, prev_keypoints=prev_kp, threshold=20
                    )
                else:
                    corrected_kp, is_valid, message = pred_kp, True, "valid"
                
                if is_valid:
                    hand_append = {'handness': hand_missing, 'keypoints': corrected_kp, 'gesture': "Track"}
                    hand_info.append(hand_append)
                    #print(f"Using newly tracked {hand_missing} hand for {imgname}: {message}")
                else:
                    print(f"Discarding newly tracked {hand_missing} hand for {imgname} due to {message}")
            else:
                print(f"Discarding newly tracked {hand_missing} hand for {imgname} due to insufficient visible keypoints ({pred_vis.sum()}/{len(pred_vis)})")
        
        # Visualize all hands (original detections + tracked hands)
        if args.vis:
            imgpath = os.path.join(img_dir, imgname)
            image = cv2.imread(imgpath)
            if image is not None:
                # Convert BGR to RGB for matplotlib
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(image_rgb)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
                
                colors = {'Left': 'red', 'Right': 'blue', 'Track': 'green'}
                
                for hand_data in hand_info:
                    keypoints = hand_data['keypoints']
                    handness = hand_data['handness']
                    gesture = hand_data.get('gesture', 'Detection')
                    
                    # Choose color based on handness and whether it's tracked
                    if gesture == 'Track':
                        color = 'green'
                        marker_size = 8
                    else:
                        color = colors.get(handness, 'yellow')
                        marker_size = 6
                    
                    # Draw keypoints as circles
                    valid_points = []
                    for i, pt in enumerate(keypoints):
                        if len(pt) >= 2 and pt[0] >= 0 and pt[1] >= 0:  # Valid keypoint
                            ax.plot(pt[0], pt[1], 'o', color=color, markersize=marker_size, alpha=0.7)
                            valid_points.append((i, pt))
                    
                    # Draw connections between keypoints
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start_pt = None
                        end_pt = None
                        
                        # Find the corresponding points in valid_points
                        for idx, pt in valid_points:
                            if idx == start_idx:
                                start_pt = pt
                            elif idx == end_idx:
                                end_pt = pt
                        
                        # Draw line if both points are valid
                        if start_pt is not None and end_pt is not None:
                            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                                   color=color, linewidth=2, alpha=0.6)
                    
                    # Add text label for handness
                    if valid_points:
                        # Use the first valid point for text placement
                        text_x, text_y = valid_points[0][1][:2]
                        label = f"{handness}" + (f" (Tracked)" if gesture == 'Track' else "")
                        ax.text(text_x, text_y - 20, label, color=color, 
                               fontsize=10, weight='bold', ha='center')
                
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, imgname), bbox_inches='tight', dpi=100)
                plt.close()
             

    with open(os.path.join(result_dir, f'handpose_{cam_name}_track.pkl'), 'wb') as file:
        pickle.dump(handpose_dct, file) 