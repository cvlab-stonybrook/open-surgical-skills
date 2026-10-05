#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import cv2
import pickle
from tqdm import tqdm
import traceback
import re
import matplotlib.pyplot as plt

# It's assumed that LightGlue is installed. If not, run: pip install lightglue
# Add the path to LightGlue if it's a local clone
# sys.path.append('/path/to/your/LightGlue')
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
except ImportError:
    print("Error: lightglue library not found.")
    print("Please install it using: pip install lightglue")
    sys.exit(1)

def frame_to_tensor(frame, device):
    """
    Convert OpenCV frame to tensor format expected by LightGlue.
    
    Args:
        frame: OpenCV frame (BGR format)
        device: torch device
    
    Returns:
        tensor: Image tensor ready for LightGlue (CHW format)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize to [0, 1]
    tensor = torch.from_numpy(frame_rgb).float() / 255.0
    # Rearrange to CHW format (no batch dimension needed)
    tensor = tensor.permute(2, 0, 1).to(device)
    return tensor

def create_dummy_matches_for_homography(kpts0, kpts1, matches):
    
    # Convert tensor keypoints to cv2.KeyPoint objects
    kp1_list = []
    kp2_list = []
    
    # Create KeyPoint objects for all keypoints
    for i, kpt in enumerate(kpts0):
        x, y = float(kpt[0]), float(kpt[1])
        kp1_list.append(cv2.KeyPoint(x, y, 1))
    
    for i, kpt in enumerate(kpts1):
        x, y = float(kpt[0]), float(kpt[1])
        kp2_list.append(cv2.KeyPoint(x, y, 1))
    
    # Create DMatch objects for matched keypoints
    match_list = []
    for i, match in enumerate(matches):
        query_idx = int(match[0])  # index in first frame
        train_idx = int(match[1])  # index in second frame
        match_list.append(cv2.DMatch(query_idx, train_idx, 0.0))
    
    return kp1_list, kp2_list, match_list

def estimate_homography(kp1, kp2, good_matches, match_threshold=6, ransac_threshold=5.0):
    """
    Estimate homography matrix from matched keypoints using RANSAC.
    
    Args:
        kp1: Keypoints from the first frame
        kp2: Keypoints from the second frame  
        good_matches: List of good matches between keypoints
        ransac_threshold: Threshold for RANSAC algorithm (default: 5.0)
    
    Returns:
        H: Homography matrix (3x3) or None if estimation fails
        mask: Mask indicating inliers from RANSAC
    """
    if len(good_matches) < match_threshold:
        print(f"Not enough matches for homography estimation: {len(good_matches)} < {match_threshold}")
        return None, None
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Estimate homography using RANSAC for robust estimation
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    
    if H is None:
        print("Homography estimation failed")
        return None, None
    
    # Count inliers
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(good_matches)
    
    #print(f"Homography estimated with {inliers}/{len(good_matches)} inliers ({inlier_ratio:.2f})")
    
    return H, mask


def homography_corner_displacement(H, frame_shape):
    """Calculate the displacement of frame corners based on a homography matrix."""
    h, w, _ = frame_shape
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    displacements = np.linalg.norm(corners - transformed_corners, axis=2).flatten()
    
    mean_disp = np.mean(displacements)
    max_disp = np.max(displacements)
    
    # Normalize by image diagonal
    diagonal = np.sqrt(w**2 + h**2)
    norm_mean_disp = mean_disp / diagonal
    norm_max_disp = max_disp / diagonal
    
    return mean_disp, max_disp, norm_mean_disp, norm_max_disp

def compute_movement_score(prev_feats, curr_feats, matcher, args):
    """Compute movement score between two frames using LightGlue."""

    with torch.no_grad():
        matches01 = matcher({"image0": prev_feats, "image1": curr_feats})
    
    # Remove batch dimension
    kpts0, kpts1, matches = [rbd(x) for x in [prev_feats, curr_feats, matches01]]
    kpts0, kpts1, matches = kpts0['keypoints'], kpts1['keypoints'], matches['matches']

    num_matches = len(matches)
    kpts0_np, kpts1_np, matches_np = [x.cpu().numpy() for x in [kpts0, kpts1, matches]]

    if num_matches < args.match_threshold:
        print(f"Insufficient matches: {num_matches}")
        return -1.0, kpts0_np, kpts1_np, matches_np # Indicate insufficient matches

    kp1_list, kp2_list, match_list = create_dummy_matches_for_homography(kpts0_np, kpts1_np, matches_np)
    
    H, _ = estimate_homography(kp1_list, kp2_list, match_list, args.match_threshold)
    
    if H is None:
        return 0.0, kpts0_np, kpts1_np, matches_np

    frame_shape = (args.frame_height, args.frame_width, 3)
    _, _, norm_mean_disp, _ = homography_corner_displacement(H, frame_shape)
    
    return norm_mean_disp, kpts0_np, kpts1_np, matches_np
        
   
def visualize_scores(vis_data, subject_name, group_name, output_dir, args, num_examples=5, vis_singleview=False):
    """Saves one vertically stacked image per high-scoring pair, showing keypoint matches."""
    if not vis_data:
        print(f"No visualization data available for {group_name}/{subject_name}.")
        return

    # Create a dedicated directory for this subject's visualizations
    subject_vis_dir = os.path.join(output_dir, 'visualizations', f'{group_name}_{subject_name}')
    os.makedirs(subject_vis_dir, exist_ok=True)
    
    # Sort vis_data to find the pairs with the highest movement scores
    #sorted_vis_data = sorted(vis_data, key=lambda x: x['score'], reverse=True)
    sorted_vis_data = vis_data
    movement_threshold = args.movement_threshold
    
    print(f"Saving top {min(num_examples, len(sorted_vis_data))} visualization images for {group_name}/{subject_name}...")

    num_vis = len(sorted_vis_data) if num_examples < 0 else num_examples
    # Process the top N frame pairs
    for i in range(num_vis):
        item = sorted_vis_data[i]
        score = item['score']
        kpts0, kpts1, matches = item['kpts0'], item['kpts1'], item['matches']
        prev_idx, curr_idx = item['prev_idx'], item['curr_idx']
        
        if not vis_singleview:
            # Load frames
            prev_frame = cv2.imread(item['prev_path'])
            curr_frame = cv2.imread(item['curr_path'])

            # Skip if images or keypoints are missing
            if prev_frame is None or curr_frame is None or kpts0 is None:
                continue
                
            # Resize frames to the processing size
            frame1 = cv2.resize(prev_frame, (args.frame_width, args.frame_height))
            frame2 = cv2.resize(curr_frame, (args.frame_width, args.frame_height))

            # Vertically concatenate the frames
            combined_frame = np.vstack([frame1, frame2])
        
            num_matches = len(matches) if matches is not None else 0

            # Draw matched keypoints and lines
            if num_matches > 0:
                for match in matches:
                    idx0, idx1 = int(match[0]), int(match[1])
                    
                    pt1 = (int(kpts0[idx0][0]), int(kpts0[idx0][1]))
                    pt2 = (int(kpts1[idx1][0]), int(kpts1[idx1][1]))
                    
                    # Adjust y-coordinate for second frame
                    pt2_adjusted = (pt2[0], pt2[1] + frame1.shape[0])
                    
                    cv2.line(combined_frame, pt1, pt2_adjusted, color=(0, 0, 255), thickness=1)
                    cv2.circle(combined_frame, pt1, 3, (255, 0, 0), -1)
                    cv2.circle(combined_frame, pt2_adjusted, 3, (255, 0, 0), -1)
            else:
                cv2.putText(combined_frame, "NO MATCHES FOUND", (10, combined_frame.shape[0] // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            # Add text labels
            y_offset = 30
            cv2.putText(combined_frame, f"Frame {prev_idx}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, f"Frame {curr_idx} (interval={args.frame_interval})", (10, frame1.shape[0] + y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add movement metrics
            y_offset += 40
            color_movement = (0, 0, 255) if score == -1 or score > movement_threshold else (255, 255, 0)
            if score == -1:
                cv2.putText(combined_frame, f"Insufficient matches: {num_matches}", (10, frame1.shape[0] + y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(combined_frame, f"Movement Score: {score:.4f}", (10, frame1.shape[0] + y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_movement, 2)
            
            y_offset += 30
            match_color = (255, 255, 255) if num_matches > 0 else (0, 0, 255)
            cv2.putText(combined_frame, f"Matches: {num_matches}", (10, frame1.shape[0] + y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, match_color, 2)
            
            # Save the visualization
            save_path = os.path.join(subject_vis_dir, f'pair_{i+1:02d}.png')
            cv2.imwrite(save_path, combined_frame)  
        else:
            curr_frame = cv2.imread(item['curr_path'])
            if curr_frame is None: continue
            frame = cv2.resize(curr_frame, (args.frame_width, args.frame_height))
            y_offset = 30
            color_movement = (0, 0, 255) if score == -1 or score > movement_threshold else (255, 255, 0)
            if score == -1:
                cv2.putText(frame, f"Insufficient matches: {num_matches}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), )
            else:
                cv2.putText(frame, f"Movement Score: {score:.4f}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_movement, 2)
            y_offset += 30
            match_color = (255, 255, 255)
            num_matches = len(matches) if matches is not None else 0
            cv2.putText(frame, f"Matches: {num_matches}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, match_color, 2)
            
            cv2.imwrite(os.path.join(subject_vis_dir, f'pair_{i+1:02d}.png'), frame)
                    
    print(f"Visualizations saved in {subject_vis_dir}")


def detect_head_movement_for_subject(subject_path, extractor, matcher, args):
    """Process all egocentric frames for a single subject to detect head movements."""
    image_dir = os.path.join(subject_path, 'Images', "Ego")
    if not os.path.isdir(image_dir):
        print(f"Warning: Egocentric image directory not found for {os.path.basename(subject_path)}. Skipping.")
        return [], []

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) < args.frame_interval + 1:
        print(f"Warning: Not enough frames for subject {os.path.basename(subject_path)}. Skipping.")
        return [], []

    scores_data = []
    visualization_data = []
    
    # Pre-extract features for the first frame
    prev_frame_path = os.path.join(image_dir, image_files[0])
    frame = cv2.imread(prev_frame_path)
    frame = cv2.resize(frame, (args.frame_width, args.frame_height))
    prev_tensor = frame_to_tensor(frame, args.device)
    prev_feats = extractor.extract(prev_tensor)

    # Iterate through the rest of the frames
    for i in tqdm(range(args.frame_interval, len(image_files), args.frame_interval), desc="Processing frames", leave=False):
        curr_frame_path = os.path.join(image_dir, image_files[i])
        frame = cv2.imread(curr_frame_path)
        if frame is None: continue
        
        frame = cv2.resize(frame, (args.frame_width, args.frame_height))
        curr_tensor = frame_to_tensor(frame, args.device)
        curr_feats = extractor.extract(curr_tensor)

        score, kpts0, kpts1, matches = compute_movement_score(prev_feats, curr_feats, matcher, args)
        
        curr_idx = int(os.path.splitext(os.path.basename(curr_frame_path))[0])
        scores_data.append({'score': score, 'frame_idx': curr_idx})
        
        # Store paths, score, and keypoint data for visualization
        if args.visualize:
            visualization_data.append({
                'score': score,
                'prev_path': prev_frame_path,
                'curr_path': curr_frame_path,
                'kpts0': kpts0,
                'kpts1': kpts1,
                'matches': matches,
                'prev_idx': int(os.path.splitext(os.path.basename(prev_frame_path))[0]),
                'curr_idx': curr_idx
            })
        
        # Current becomes previous for the next iteration
        prev_feats = curr_feats
        prev_frame_path = curr_frame_path
    
    return scores_data, visualization_data

def main(args):
    """Main function to iterate subjects, compute head movement, and save results."""
    data_dir = os.path.join(args.base_dir, 'Data')
    output_dir = os.path.join(args.base_dir, 'Analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    group_names = ['Students', 'Residents', 'Attendings']
    head_movement_scores = {group: {} for group in group_names}
    
    # Initialize models
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"Using device: {device}")
    
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    try:
        for group in group_names:
            group_dir = os.path.join(data_dir, group)
            if not os.path.isdir(group_dir):
                print(f"Group directory not found: {group_dir}. Skipping.")
                continue
            
            pattern = re.compile(r'^\d+')
            subject_folders = sorted([f for f in os.listdir(group_dir) if pattern.match(f) and os.path.isdir(os.path.join(group_dir, f))])
            
            print(f"\nProcessing group: {group}")
            for subject in tqdm(subject_folders, desc=f"Subjects in {group}"):
                subject_path = os.path.join(group_dir, subject)
                
                scores_data, vis_data = detect_head_movement_for_subject(subject_path, extractor, matcher, args)
                head_movement_scores[group][subject] = scores_data

                # Visualize if requested
                if args.visualize and (args.vis_subject is None or f"{group}/{subject}" == args.vis_subject):
                    visualize_scores(vis_data, subject, group, output_dir, args, num_examples=args.vis_examples, vis_singleview=args.vis_singleview)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    # Save results to pickle
    output_path = os.path.join(output_dir, 'head_movement_scores_ego.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(head_movement_scores, f)
            
    print(f"\nProcessing complete. Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect large head movements in egocentric videos.")
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="Base directory containing 'Data' and where 'Analysis' will be saved.")
    
    # Head Movement Parameters
    parser.add_argument('--frame_interval', type=int, default=1, help='Frame interval for movement computation.')
    parser.add_argument('--match_threshold', type=int, default=20, help='Minimum feature matches to consider a valid transform.')
    parser.add_argument('--movement_threshold', type=float, default=0.03, help='Movement score threshold to count as a large movement (only used for visualization here).')
    # NOTE: the frames are resized to (frame_width, frame_height) = (360, 640) regardless of their aspect ratio.
    # These are the values used for the results in the paper.
    parser.add_argument('--frame_height', type=int, default=640, help='Frame height for processing.')
    parser.add_argument('--frame_width', type=int, default=360, help='Frame width for processing.')

    # LightGlue/SuperPoint Parameters
    parser.add_argument('--device', default='0', help='GPU device ID.')
    parser.add_argument('--max_keypoints', type=int, default=1024, help='Maximum keypoints to extract per frame.')
    parser.add_argument('--vis_singleview', action='store_true', help='Visualize single view frames.')
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Generate and save plots of movement scores.')
    parser.add_argument('--vis_subject', type=str, default=None, help='Specific subject to visualize (e.g., "Students/01_30_01"). If not set, all subjects are plotted.')
    parser.add_argument('--vis_examples', type=int, default=-1, help='Number of frame pair examples to save as images.')

    args = parser.parse_args()
    main(args)

