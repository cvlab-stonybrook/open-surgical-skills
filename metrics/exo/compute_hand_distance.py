import os
import numpy as np
import pandas as pd
import argparse
import pickle
import re
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, required=True, help='dataset root containing the Data and Analysis folders')
parser.add_argument("--fps", type=int, default=5, help='frame rate of the frame indices in the image names (5 if the frames are decoded with scripts/decode_frames.sh)')
parser.add_argument("--abnormal_thresh", type=int, default=40, help='distance above this threshold is considered abnormal, highly likely that the hand in this frame is an outlier from noisy estimation')
parser.add_argument("--noise_thresh", type=float, default=0.0, help='distance below this threshold is considered noise and will be ignored')
args = parser.parse_args()

# Set up logging file

data_dir = os.path.join(args.base_dir, 'Data')
group_names = ['Students', 'Residents', 'Attendings']
excel_path = os.path.join(args.base_dir, 'Analysis', 'cycle_annotations.xlsx')
df = pd.read_excel(excel_path, header=0, index_col=0)
subj_names = df.index.tolist()

# Read left-handed subjects list
left_dominant_file = os.path.join(args.base_dir, 'Analysis', 'left_dominant_list.txt')
left_handed_subjects = set()
if os.path.exists(left_dominant_file):
    with open(left_dominant_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ',' in line:
                group, subj_name = line.split(',', 1)
                left_handed_subjects.add((group.strip(), subj_name.strip()))
    print(f"Loaded {len(left_handed_subjects)} left-handed subjects from {left_dominant_file}")
else:
    print(f"Warning: {left_dominant_file} not found. Assuming all subjects are right-handed.")

fps = args.fps
hand_dist = {'Students':{}, 'Residents':{}, 'Attendings':{}}
abnormal_thresh = args.abnormal_thresh
dist_left_frame, dist_right_frame = [], []

# New variable to store keypoints by cycles
keypoints_by_cycles = {'Students':{}, 'Residents':{}, 'Attendings':{}}

try:
    for group in group_names:
        group_dir = os.path.join(data_dir, group)
        pattern = re.compile(r'^\d+')
        subj_names = sorted([folder for folder in os.listdir(group_dir) if pattern.match(folder)])
        for subj_folder in subj_names:
            #if subj_folder != '09_30':
            #    continue
            print("subject %s" % subj_folder)
            # Reset per-subject frame distance buffers
            dist_left_frame, dist_right_frame = [], []
            
            log_file = os.path.join(data_dir, group, subj_folder, 'handpose', 'vis_3d', 'hand_distance.txt')
            
            cycles_subj = df.loc[group+'_'+subj_folder]
            cycles_subj = cycles_subj.dropna().values[::2]
            cycles_this = np.array( [ int(cycle.split(':')[0]) * 60 + int(cycle.split(':')[1]) for cycle in cycles_subj] )
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f"{subj_folder}\n")
            frame_idx = cycles_this * fps
            base_dir = os.path.join(data_dir, group, subj_folder)
            hand_3d_path_default = os.path.join(base_dir, 'handpose', 'keypoints_3d.pickle')
            hand_3d_path_smoothed = os.path.join(base_dir, 'handpose', 'keypoints_3d_smoothed.pickle')
            hand_3d_path = hand_3d_path_smoothed if os.path.exists(hand_3d_path_smoothed) else hand_3d_path_default
            with open(hand_3d_path, 'rb') as file:
                kp_3d = pickle.load(file)

            hand_dist[group][subj_folder] = {'left':[], 'right':[]}
            # Initialize keypoints by cycles for this subject
            keypoints_by_cycles[group][subj_folder] = []
            
            kp3d_l, kp3d_r, kp_valid_l, kp_valid_r, imagelist = kp_3d['left'], kp_3d['right'], kp_3d['left_valid'], kp_3d['right_valid'], kp_3d['image']

            assert len(kp3d_l) == len(kp3d_r)
            assert len(kp3d_l) == len(kp_valid_l)

            cycle_idx = 0
            # Initialize cycle keypoints
            cycle_keypoints = {
                'left': [],
                'right': [],
                'left_valid': [],
                'right_valid': [],
                'image': []
            }
            last_idx_left, last_idx_right = 0,0
            for idx, kp_l in enumerate(kp3d_l):
                img_idx = int(imagelist[idx].split('.')[0])
                if img_idx < frame_idx[0]:
                    continue
                if img_idx > frame_idx[-1]:
                    break
                
                if img_idx >= frame_idx[cycle_idx]:
                    
                    if cycle_idx != 0:
                        hand_dist[group][subj_folder]['left'].append(hand_dist_l)
                        hand_dist[group][subj_folder]['right'].append(hand_dist_r)
                        #if subj_folder=='01_30_01':
                        #    print(subj_folder, f"cycle_idx: {cycle_idx}, hand_dist_l: {hand_dist_l}, hand_dist_r: {hand_dist_r}")
                        
                        # Save the previous cycle keypoints
                        keypoints_by_cycles[group][subj_folder].append(cycle_keypoints)
                        # Initialize new cycle keypoints
                        cycle_keypoints = {
                            'left': [],
                            'right': [],
                            'left_valid': [],
                            'right_valid': [],
                            'image': []
                        }
                        # if subj_folder=='09_30':
                        #     print(f"********* cycle_idx: {cycle_idx} *********")
                        #     print(f"Left: {dist_left_frame[last_idx_left:]},\n Right: {dist_right_frame[last_idx_right:]}")
                        #     #print(f"num frames: {len(dist_left_frame) - last_idx_left} Left: {hand_dist_l}, Right: {hand_dist_r}")
                        #     last_idx_left, last_idx_right = len(dist_left_frame), len(dist_right_frame)

                    hand_center_lst_l, hand_center_lst_r = [], []
                    hand_center_l_prev, hand_center_r_prev = np.zeros(3), np.zeros(3)
                    hand_dist_l, hand_dist_r = 0,0  
                    cycle_idx += 1
                
                kp_l, kp_r, valid_l, valid_r = kp3d_l[idx], kp3d_r[idx], kp_valid_l[idx], kp_valid_r[idx]
                
                valid_joints_l = kp_l[:, 3] > 0
                hand_center_l = (kp_l[valid_joints_l, :3].mean(axis=0) if valid_joints_l.any() else hand_center_l_prev)
                if valid_l and valid_joints_l.any():
                    if not np.all(hand_center_l_prev==0):
                        this_dist = np.linalg.norm(hand_center_l - hand_center_l_prev)
                        
                        if this_dist > abnormal_thresh:
                            dist_to_2before = np.linalg.norm(hand_center_l - hand_center_lst_l[-2]) if len(hand_center_lst_l) >= 2 else float('inf')
                            if len(dist_left_frame) > 0 and dist_left_frame[-1][1] > abnormal_thresh and dist_to_2before <= abnormal_thresh//2:
                                # consider the last frame as an outlier, and replace it with the average of the current and 2 frame before
                                handpose_l_2before = kp3d_l[idx-2]
                                handpose_l_1before = kp3d_l[idx-1]
                                handpose_l_current = kp3d_l[idx]
                                handpose_l_avg = (handpose_l_2before + handpose_l_current) / 2
                                print("Correct for left hand at frame %s" % imagelist[idx-1])
                                kp3d_l[idx-1] = handpose_l_avg
                                cycle_keypoints['left'][-1] = handpose_l_avg
                                hand_center_lst_l[-1] = (hand_center_lst_l[-2] + hand_center_l) / 2 if len(hand_center_lst_l) >= 2 else hand_center_l
                                hand_dist_l += dist_to_2before if np.isfinite(dist_to_2before) else 0
                                dist_left_frame[-1] = (dist_left_frame[-1][0], (dist_to_2before / 2) if np.isfinite(dist_to_2before) else dist_left_frame[-1][1])
                        else:
                            if this_dist > args.noise_thresh:
                                hand_dist_l += this_dist
                        dist_left_frame.append((img_idx, this_dist))
                
                hand_center_lst_l.append(hand_center_l)                               
                hand_center_l_prev = hand_center_l    
                    
                    
                valid_joints_r = kp_r[:, 3] > 0
                hand_center_r = (kp_r[valid_joints_r, :3].mean(axis=0) if valid_joints_r.any() else hand_center_r_prev)
                if valid_r and valid_joints_r.any():
                    if not np.all(hand_center_r_prev==0):
                        this_dist = np.linalg.norm(hand_center_r - hand_center_r_prev)
                        if this_dist > abnormal_thresh:
                            dist_to_2before = np.linalg.norm(hand_center_r - hand_center_lst_r[-2]) if len(hand_center_lst_r) >= 2 else float('inf')
                            if len(dist_right_frame) > 0 and dist_right_frame[-1][1] > abnormal_thresh and dist_to_2before <= abnormal_thresh:
                                # consider the last frame as an outlier, and replace it with the average of the current and 2 frame before
                                handpose_r_2before = kp3d_r[idx-2]
                                handpose_r_1before = kp3d_r[idx-1]
                                handpose_r_current = kp3d_r[idx]
                                handpose_r_avg = (handpose_r_2before + handpose_r_current) / 2
                                print("Correct for right hand at frame %s" % imagelist[idx-1])
                                kp3d_r[idx-1] = handpose_r_avg
                                cycle_keypoints['right'][-1] = handpose_r_avg
                                hand_center_lst_r[-1] = (hand_center_lst_r[-2] + hand_center_r) / 2
                                hand_dist_r += dist_to_2before
                                dist_right_frame[-1] = (dist_right_frame[-1][0], dist_to_2before / 2)
                        else:
                            if this_dist > args.noise_thresh:
                                hand_dist_r += this_dist
                        dist_right_frame.append((img_idx, this_dist))
                hand_center_lst_r.append(hand_center_r)                               
                hand_center_r_prev = hand_center_r

                # Store keypoints for this frame in the current cycle
                cycle_keypoints['left'].append(kp_l)
                cycle_keypoints['right'].append(kp_r)
                cycle_keypoints['left_valid'].append(valid_l)
                cycle_keypoints['right_valid'].append(valid_r)
                cycle_keypoints['image'].append(imagelist[idx])
            
            # Save the last cycle keypoints
            if cycle_keypoints['left']:  # Only save if there are keypoints
                keypoints_by_cycles[group][subj_folder].append(cycle_keypoints)
                hand_dist[group][subj_folder]['left'].append(hand_dist_l)
                hand_dist[group][subj_folder]['right'].append(hand_dist_r)
                
                
            
            for img_idx, dist in dist_right_frame:
                cycle_idx = 0
                if img_idx < frame_idx[0]:
                    continue
                if img_idx > frame_idx[-1]:
                    break
                if img_idx >= frame_idx[cycle_idx]:
                    with open(log_file, 'a') as f:
                        f.write(f"Cycle {cycle_idx}\n")
                    cycle_idx += 1
                with open(log_file, 'a') as f:
                    f.write(f" {img_idx} {dist}\n")
            with open(os.path.join(base_dir, 'handpose', 'keypoints_3d_filtered.pickle'), 'wb') as file:
                pickle.dump(kp_3d, file)
            
except Exception as e:
    print(traceback.format_exc())
    raise

# write average cycle distance as excel file:
with pd.ExcelWriter(os.path.join(args.base_dir, 'Analysis', f'hand_dists_allgroups_smoothed_{args.noise_thresh}.xlsx')) as writer:
    for group in group_names:
        df = []
        for subj_folder in hand_dist[group]:
            left_avg = np.mean(hand_dist[group][subj_folder]['left'])
            right_avg = np.mean(hand_dist[group][subj_folder]['right'])
            #df.append([subj_folder, round(left_avg, ndigits=2), round(right_avg, ndigits=2)])
        #df = pd.DataFrame(df, columns=['Subject', 'Left_Distance', 'Right_Distance'])
            
            # Check if subject is left-handed and swap dominant/non-dominant assignment
            is_left_handed = (group, subj_folder) in left_handed_subjects
            if is_left_handed:
                # For left-handed: left hand is dominant, right hand is non-dominant
                dominant_dist = left_avg
                nondominant_dist = right_avg
                print(f"Left-handed subject detected: {group}/{subj_folder} - Left (dominant): {left_avg:.2f}, Right (non-dominant): {right_avg:.2f}")
            else:
                # For right-handed: right hand is dominant, left hand is non-dominant
                dominant_dist = right_avg
                nondominant_dist = left_avg
            
            df.append([subj_folder, round(nondominant_dist, ndigits=2), round(dominant_dist, ndigits=2)])
        df = pd.DataFrame(df, columns=['Subject', 'NonDominant_Distance', 'Dominant_Distance'])
        df.to_excel(writer, sheet_name=group, header=True, index=False)
    

 
with open(os.path.join(args.base_dir, 'Analysis', f'hand_distance_allgroups_smoothed_{args.noise_thresh}.pkl'), 'wb') as file:
    pickle.dump(hand_dist, file)    

# Save keypoints by cycles
with open(os.path.join(args.base_dir, 'Analysis', 'keypoints_by_cycles_smoothed.pkl'), 'wb') as file:
    pickle.dump(keypoints_by_cycles, file)
