import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.spatial import procrustes
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_dtw_similarity(trajectories):
    M = len(trajectories)  # Number of cycles
    total_dtw_distance = 0
    count = 0

    for i in range(M):
        for j in range(i + 1, M):
            #traj_1, traj_2 = trajectories[i][:, :3], trajectories[j][:, :3]
            dist, _ = fastdtw(trajectories[i], trajectories[j], dist=euclidean)
            total_dtw_distance += dist
            count += 1

    # Normalize the DTW similarity
    avg_dtw_distance = total_dtw_distance / count
    return 1 / (1 + avg_dtw_distance)  # Convert to similarity score (higher is more stable)


def compute_procrustes_similarity(trajectories):
    # please note that procustes assumes that the two trajectories have the same number of points
    M = len(trajectories)  # Number of cycles
    total_procrustes_dist = 0
    count = 0

    for i in range(M):
        for j in range(i + 1, M):
            traj_1, traj_2 = trajectories[i], trajectories[j]
            if len(trajectories[i]) != len(trajectories[j]):
                # uniformly sample the longer trajectory to the same length as the shorter one
                if len(trajectories[i]) > len(trajectories[j]):
                    # Create indices for interpolation
                    x = np.arange(len(trajectories[i]))
                    x_new = np.linspace(0, len(trajectories[i])-1, len(trajectories[j]))
                    # Interpolate each dimension
                    traj_i = np.zeros((len(trajectories[j]), trajectories[i].shape[1]))
                    for dim in range(trajectories[i].shape[1]):
                        traj_i[:, dim] = np.interp(x_new, x, trajectories[i][:, dim])
                    traj_1 = traj_i
                else:
                    # Create indices for interpolation
                    x = np.arange(len(trajectories[j]))
                    x_new = np.linspace(0, len(trajectories[j])-1, len(trajectories[i]))
                    # Interpolate each dimension
                    traj_j = np.zeros((len(trajectories[i]), trajectories[j].shape[1]))
                    for dim in range(trajectories[j].shape[1]):
                        traj_j[:, dim] = np.interp(x_new, x, trajectories[j][:, dim])
                    traj_2 = traj_j
            mtx1, mtx2, disparity = procrustes(traj_1, traj_2)
            total_procrustes_dist += disparity
            count += 1

    # Normalize Procrustes similarity (1 - average Procrustes distance)
    avg_procrustes_dist = total_procrustes_dist / count
    return 1 - avg_procrustes_dist  # Higher values indicate more stability


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, required=True, help='dataset root containing the Data and Analysis folders')
args = parser.parse_args()

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

with open(os.path.join(args.base_dir, 'Analysis', 'keypoints_by_cycles_smoothed.pkl'), 'rb') as file:
    cycle_keypoints = pickle.load(file)

scores_all = {}
for group, subj_dict in cycle_keypoints.items():
    scores_all[group] = {}
    for subj, cycle_list in tqdm(subj_dict.items()):
        print(subj)
        all_cycles_left, all_cycles_right = [], []
        for cycle in cycle_list:
            cycle_center_left, cycle_center_right = [], []
            for kp_left, kp_right in zip(cycle['left'], cycle['right']):
                center_left, center_right = np.mean(kp_left, axis=0)[:-1], np.mean(kp_right, axis=0)[:-1]
                cycle_center_left.append(center_left)
                cycle_center_right.append(center_right)
            cycle_center_left = np.array(cycle_center_left)
            cycle_center_right = np.array(cycle_center_right)
            if len(cycle_center_left) > 10 and len(cycle_center_right) > 10:
                all_cycles_left.append(cycle_center_left)
                all_cycles_right.append(cycle_center_right)
                
        #scores_all[group][subj] = {'left': compute_dtw_similarity(all_cycles_left), 'right': compute_dtw_similarity(all_cycles_right)}
        scores_all[group][subj] = {'left': compute_procrustes_similarity(all_cycles_left), 'right': compute_procrustes_similarity(all_cycles_right)}
# Write scores to Excel file
# Create a Pandas Excel writer using XlsxWriter as the engine
output_path = os.path.join(args.base_dir, 'Analysis', 'hand_stability_scores_procrustes_smoothed.xlsx')
with pd.ExcelWriter(output_path) as writer:
    # For each group, create a DataFrame and write to a separate sheet
    for group, subj_dict in scores_all.items():
        # Create a list of dictionaries for each subject
        data = []
        for subj, scores in subj_dict.items():
            left_score = scores['left']
            right_score = scores['right']
            
            # Check if subject is left-handed and swap dominant/non-dominant assignment
            is_left_handed = (group, subj) in left_handed_subjects
            if is_left_handed:
                # For left-handed: left hand is dominant, right hand is non-dominant
                dominant_score = left_score
                nondominant_score = right_score
                print(f"Left-handed subject detected: {group}/{subj} - Left (dominant): {left_score:.4f}, Right (non-dominant): {right_score:.4f}")
            else:
                # For right-handed: right hand is dominant, left hand is non-dominant
                dominant_score = right_score
                nondominant_score = left_score
            
            data.append({
                'Subject': subj,
                'NonDominant_Hand_Stability': nondominant_score,
                'Dominant_Hand_Stability': dominant_score
            })
        
        # Create DataFrame and write to Excel
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=group, index=False, header=False)
        
        
print(f"Hand stability scores have been saved to {output_path}")





