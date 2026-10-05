import os
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

def calculate_sparc(trajectory, fs, cutoff=10.0):
    """
    Calculates the SPARC (Spectral Arc Length) smoothness metric.
    This implementation returns a value where a higher score means smoother movement.

    Args:
        trajectory (np.array): A numpy array of shape (n_frames, n_dims)
                               representing the movement trajectory.
        fs (int): The sampling frequency of the trajectory.
        cutoff (float): The frequency cutoff for the calculation. Human movements
                        typically don't exceed 10Hz.

    Returns:
        float: The SPARC value. Returns 0 if the trajectory is too short or has no movement.
    """
    # Trajectory must have at least 2 points to calculate velocity
    if trajectory.shape[0] < 2:
        return 0.0

    # 1. Calculate velocity from the position data
    # np.diff computes the difference between adjacent elements
    velocity = np.diff(trajectory, axis=0) * fs
    
    # Velocity must have at least 2 points to calculate the spectrum properly
    if velocity.shape[0] < 2:
        return 0.0

    # 2. Calculate the Fourier Transform of the velocity
    n_frames = velocity.shape[0]
    freq = np.fft.fftfreq(n_frames, 1/fs)
    
    # We only care about the positive frequencies up to the cutoff
    valid_indices = np.where((freq >= 0) & (freq <= cutoff))[0]
    freq_sel = freq[valid_indices]
    
    # Calculate the magnitude spectrum for each dimension (x, y, z) and sum them up
    mag_spectrum_sel_sum = np.zeros_like(freq_sel, dtype=float)
    
    for dim in range(trajectory.shape[1]):
        fft_vel = np.fft.fft(velocity[:, dim])
        mag_spectrum = np.abs(fft_vel)
        mag_spectrum_sel = mag_spectrum[valid_indices]
        mag_spectrum_sel_sum += mag_spectrum_sel

    # Average the magnitude spectrum across all dimensions
    mag_spectrum_sel_avg = mag_spectrum_sel_sum / trajectory.shape[1]

    # 3. Normalize the magnitude spectrum so its peak is 1
    peak_magnitude = np.max(mag_spectrum_sel_avg)
    if peak_magnitude == 0:
        return 0.0 # No movement, so perfectly smooth
    
    mag_norm = mag_spectrum_sel_avg / peak_magnitude

    # 4. Calculate the Spectral Arc Length
    # This is the integral of sqrt(1 + (dV/df)^2) df, approximated discretely.
    # A smoother movement has a more compact spectrum, resulting in a smaller arc length.
    delta_f = freq_sel[1] - freq_sel[0] if len(freq_sel) > 1 else 1
    deriv_mag = np.diff(mag_norm) / delta_f
    arc_length = np.sum(np.sqrt(1 + deriv_mag**2) * delta_f)
    
    
    # a lower arc_length indicates more smoothness.
    return arc_length


def main(args):
    """
    Main function to load data, compute smoothness, and save results.
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

    # Load the keypoints data you generated previously
    keypoints_file = os.path.join(args.base_dir, 'Analysis', 'keypoints_by_cycles_smoothed.pkl')
    with open(keypoints_file, 'rb') as file:
        cycle_keypoints = pickle.load(file)

    scores_all = {}
    group_names = ['Students', 'Residents', 'Attendings']

    for group in group_names:
        if group not in cycle_keypoints:
            continue
        
        subj_dict = cycle_keypoints[group]
        scores_all[group] = {}
        
        # Determine FPS based on the group
        fps = args.fps
        
        print(f"\nProcessing group: {group} with FPS={fps}")
        for subj, cycle_list in tqdm(subj_dict.items(), desc=f"Subjects in {group}"):
            
            sparc_scores_left, sparc_scores_right = [], []
            
            for cycle in cycle_list:
                # Extract hand center trajectories for the cycle
                # We take the mean of all keypoints for each hand at each frame
                cycle_center_left = np.array([np.mean(kp_left[:, :3], axis=0) for kp_left in cycle['left']])
                cycle_center_right = np.array([np.mean(kp_right[:, :3], axis=0) for kp_right in cycle['right']])

                # Ensure the trajectory is long enough for meaningful analysis
                if cycle_center_left.shape[0] > 10:
                    score_left = calculate_sparc(cycle_center_left, fs=fps)
                    sparc_scores_left.append(score_left)
                
                if cycle_center_right.shape[0] > 10:
                    score_right = calculate_sparc(cycle_center_right, fs=fps)
                    sparc_scores_right.append(score_right)
            
            # Average the scores across all valid cycles for the subject
            avg_score_left = np.mean(sparc_scores_left) if sparc_scores_left else 0.0
            avg_score_right = np.mean(sparc_scores_right) if sparc_scores_right else 0.0
            
            scores_all[group][subj] = {'left': avg_score_left, 'right': avg_score_right}

    # Write scores to an Excel file
    output_path = os.path.join(args.base_dir, 'Analysis', 'hand_smoothness_scores_sparc.xlsx')
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
                    'NonDominant_Hand_Smoothness': nondominant_score,
                    'Dominant_Hand_Smoothness': dominant_score
                })
            
            df = pd.DataFrame(data)
            
            # Calculate and print average scores for the group
            avg_nondominant = df['NonDominant_Hand_Smoothness'].mean()
            avg_dominant = df['Dominant_Hand_Smoothness'].mean()
            print(f"\nAverage scores for {group}:")
            print(f"  - Non-dominant Hand Smoothness: {avg_nondominant:.4f}")
            print(f"  - Dominant Hand Smoothness: {avg_dominant:.4f}")

            df.to_excel(writer, sheet_name=group, index=False)
            
    print(f"\nHand smoothness scores (SPARC) have been saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=5, help='frame rate of the pose sequences')
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="The base directory where your 'Analysis' folder is located.")
    args = parser.parse_args()
    main(args)
