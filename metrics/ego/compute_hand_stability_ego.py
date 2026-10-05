import os
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.spatial import procrustes
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_procrustes_similarity(trajectories):
    """
    Computes Procrustes similarity between a list of trajectories.
    Handles trajectories of different lengths via interpolation.
    """
    if len(trajectories) < 2:
        return 0.0  # Cannot compute similarity for less than 2 trajectories

    M = len(trajectories)
    total_procrustes_dist = 0
    count = 0

    for i in range(M):
        for j in range(i + 1, M):
            traj_1_orig, traj_2_orig = trajectories[i], trajectories[j]
            
            # Ensure trajectories are long enough
            if traj_1_orig.shape[0] < 2 or traj_2_orig.shape[0] < 2:
                continue

            # Standardize trajectory length via interpolation
            if traj_1_orig.shape[0] != traj_2_orig.shape[0]:
                if traj_1_orig.shape[0] > traj_2_orig.shape[0]:
                    shorter, longer = traj_2_orig, traj_1_orig
                else:
                    shorter, longer = traj_1_orig, traj_2_orig
                
                x = np.arange(longer.shape[0])
                x_new = np.linspace(0, longer.shape[0] - 1, shorter.shape[0])
                
                resampled_longer = np.zeros_like(shorter)
                for dim in range(longer.shape[1]):
                    resampled_longer[:, dim] = np.interp(x_new, x, longer[:, dim])
                
                if traj_1_orig.shape[0] > traj_2_orig.shape[0]:
                    traj_1, traj_2 = resampled_longer, shorter
                else:
                    traj_1, traj_2 = shorter, resampled_longer
            else:
                traj_1, traj_2 = traj_1_orig, traj_2_orig

            try:
                mtx1, mtx2, disparity = procrustes(traj_1, traj_2)
                total_procrustes_dist += disparity
                count += 1
            except ValueError:
                print("Warning: Procrustes analysis failed for a pair of trajectories.")
                continue

    if count == 0:
        return 0.0

    # Normalize Procrustes similarity (1 - average Procrustes distance)
    avg_procrustes_dist = total_procrustes_dist / count
    return 1 - avg_procrustes_dist  # Higher values indicate more stability


def main(args):
    """
    Main function to load egocentric 2D data, compute stability, and save results.
    """
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

    # Load keypoints data
    keypoints_file = os.path.join(args.base_dir, 'Analysis', args.input_file)
    with open(keypoints_file, 'rb') as file:
        cycle_keypoints = pickle.load(file)

    scores_all = {}
    group_names = ['Students', 'Residents', 'Attendings']

    for group in group_names:
        if group not in cycle_keypoints:
            continue
        
        subj_dict = cycle_keypoints[group]
        scores_all[group] = {}
        
        print(f"\nProcessing group: {group}")
        for subj, cycle_list in tqdm(subj_dict.items(), desc=f"Subjects in {group}"):
            all_cycles_left, all_cycles_right = [], []
            
            for cycle in cycle_list:
                # Extract and process left hand trajectory for the cycle
                kps_left = cycle.get('left', [])
                valid_left = cycle.get('left_valid', [])
                cycle_center_left = np.array([np.mean(kp[:, :2], axis=0) for kp, is_valid in zip(kps_left, valid_left) if is_valid and kp.size > 0])
                if cycle_center_left.shape[0] > 10:
                    all_cycles_left.append(cycle_center_left)
                
                # Extract and process right hand trajectory for the cycle
                kps_right = cycle.get('right', [])
                valid_right = cycle.get('right_valid', [])
                cycle_center_right = np.array([np.mean(kp[:, :2], axis=0) for kp, is_valid in zip(kps_right, valid_right) if is_valid and kp.size > 0])
                if cycle_center_right.shape[0] > 10:
                    all_cycles_right.append(cycle_center_right)
            
            # Compute stability scores
            score_left = compute_procrustes_similarity(all_cycles_left)
            score_right = compute_procrustes_similarity(all_cycles_right)
            
            scores_all[group][subj] = {'left': score_left, 'right': score_right}

    # Write scores to Excel file
    output_path = os.path.join(args.base_dir, 'Analysis', f'hand_stability_scores_{args.output_suffix}.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        for group, subj_dict in scores_all.items():
            data = []
            for subj, scores in subj_dict.items():
                left_score = scores['left']
                right_score = scores['right']
                
                # Assign scores to dominant/non-dominant hands
                is_left_handed = (group, subj) in left_handed_subjects
                if is_left_handed:
                    dominant_score = left_score
                    nondominant_score = right_score
                else:
                    dominant_score = right_score
                    nondominant_score = left_score
                
                data.append({
                    'Subject': subj,
                    'NonDominant_Hand_Stability': nondominant_score,
                    'Dominant_Hand_Stability': dominant_score
                })
            
            df = pd.DataFrame(data)
            
            # Calculate and print average scores for the group
            if not df.empty:
                avg_nondominant = df['NonDominant_Hand_Stability'].mean()
                avg_dominant = df['Dominant_Hand_Stability'].mean()
                print(f"\nAverage scores for {group}:")
                print(f"  - Non-dominant Hand Stability: {avg_nondominant:.4f}")
                print(f"  - Dominant Hand Stability: {avg_dominant:.4f}")

            df.to_excel(writer, sheet_name=group, index=False)
            
    print(f"\nHand stability scores (Procrustes) have been saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute hand movement stability (Procrustes) from egocentric 2D keypoints.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="The base directory where your 'Analysis' folder is located.")
    parser.add_argument("--input_file", type=str, default='keypoints_by_cycles_ego.pkl',
                        help="The name of the keypoints pickle file in the 'Analysis' folder.")
    parser.add_argument("--output_suffix", type=str, default='procrustes_ego',
                        help="Suffix to add to the output Excel file name.")
    args = parser.parse_args()
    main(args)
