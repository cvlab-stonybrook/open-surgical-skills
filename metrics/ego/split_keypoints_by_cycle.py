import os
import numpy as np
import pandas as pd
import argparse
import pickle
import re
import traceback
from tqdm import tqdm

def main(args):
    """
    Main function to load egocentric keypoint data, segment it by cycles, and save the result.
    """
    data_dir = os.path.join(args.base_dir, 'Data')
    group_names = ['Students', 'Residents', 'Attendings']
    excel_path = os.path.join(args.base_dir, 'Analysis', 'cycle_annotations.xlsx')
    
    try:
        df = pd.read_excel(excel_path, header=0, index_col=0)
    except FileNotFoundError:
        print(f"ERROR: Cycle annotation file not found at {excel_path}")
        return

    # This dictionary will store the final segmented keypoints
    keypoints_by_cycles = {'Students': {}, 'Residents': {}, 'Attendings': {}}

    try:
        for group in group_names:
            fps = args.fps
            group_dir = os.path.join(data_dir, group)
            if not os.path.isdir(group_dir):
                continue

            pattern = re.compile(r'^\d+')
            subj_names = sorted([folder for folder in os.listdir(group_dir) if pattern.match(folder)])
            
            
            print(f"\nProcessing group: {group}")
            for subj_folder in tqdm(subj_names, desc=f"Subjects in {group}"):
                
                # --- Get Cycle Timings ---
                try:
                    cycles_subj = df.loc[group + '_' + subj_folder]
                    cycles_subj = cycles_subj.dropna().values[::2]
                    cycles_this = np.array([int(c.split(':')[0]) * 60 + int(c.split(':')[1]) for c in cycles_subj])
                    frame_idx = cycles_this * fps
                except KeyError:
                    print(f"  - WARNING: Could not find cycle annotations for {group}/{subj_folder}. Skipping.")
                    continue

                # --- Load Egocentric Keypoint Data ---
                base_dir_subj = os.path.join(data_dir, group, subj_folder)
                hand_2d_path_default = os.path.join(base_dir_subj, 'handpose', 'handpose_Ego.pkl')
                hand_2d_path_smoothed = os.path.join(base_dir_subj, 'handpose', 'handpose_Ego_smoothed.pkl')
                hand_2d_path = hand_2d_path_smoothed if os.path.exists(hand_2d_path_smoothed) else hand_2d_path_default

                if not os.path.exists(hand_2d_path):
                    continue
                
                with open(hand_2d_path, 'rb') as file:
                    kp_2d = pickle.load(file)

                keypoints_by_cycles[group][subj_folder] = []
                
                kp2d_l = kp_2d.get('left', [])
                kp2d_r = kp_2d.get('right', [])
                imagelist = kp_2d.get('image', [])
                valid_l = kp_2d.get('left_valid', [])
                valid_r = kp_2d.get('right_valid', [])

                if not imagelist or (not kp2d_l and not kp2d_r):
                    continue
                
                # --- Segment Keypoints by Cycle ---
                cycle_idx = 0
                cycle_keypoints = {'left': [], 'right': [], 'left_valid': [], 'right_valid': []}

                for idx, img_name in enumerate(imagelist):
                    # Extract frame number from image filename (e.g., '000123.jpg')
                    img_idx_str = os.path.splitext(os.path.basename(img_name))[0]
                    if not img_idx_str.isdigit(): continue
                    img_idx = int(img_idx_str)
                    
                    if img_idx < frame_idx[0]:
                        continue
                    if img_idx > frame_idx[-1]:
                        break
                    
                    # Check if we have moved to a new cycle
                    if cycle_idx < len(frame_idx) and img_idx >= frame_idx[cycle_idx]:
                        if cycle_idx > 0:
                            # Save the completed cycle's data
                            if cycle_keypoints['left'] or cycle_keypoints['right']:
                                keypoints_by_cycles[group][subj_folder].append(cycle_keypoints)
                        
                        # Initialize storage for the new cycle
                        cycle_keypoints = {'left': [], 'right': [], 'left_valid': [], 'right_valid': []}
                        cycle_idx += 1
                    
                    # Append keypoints for the current frame
                    if idx < len(kp2d_l):
                        cycle_keypoints['left'].append(kp2d_l[idx])
                    if idx < len(kp2d_r):
                        cycle_keypoints['right'].append(kp2d_r[idx])
                    if idx < len(valid_l):
                        cycle_keypoints['left_valid'].append(valid_l[idx])
                    if idx < len(valid_r):
                        cycle_keypoints['right_valid'].append(valid_r[idx])

                # Save the last cycle's data after the loop finishes
                if cycle_keypoints['left'] or cycle_keypoints['right']:
                    keypoints_by_cycles[group][subj_folder].append(cycle_keypoints)
                    print(f"Saved {len(keypoints_by_cycles[group][subj_folder])} cycles for {group}/{subj_folder}")

    except Exception:
        print(traceback.format_exc())

    # --- Save the final dictionary to a new pickle file ---
    output_filename = 'keypoints_by_cycles_ego.pkl'
    output_path = os.path.join(args.base_dir, 'Analysis', output_filename)
    with open(output_path, 'wb') as file:
        pickle.dump(keypoints_by_cycles, file)
    print(f"\nEgocentric keypoints by cycles have been saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segment egocentric hand keypoints by surgical task cycles.")
    parser.add_argument("--fps", type=int, default=5, help='frame rate of the frame indices in the image names (5 if the frames are decoded with scripts/decode_frames.sh)')
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="The base directory where your 'Data' and 'Analysis' folders are located.")
    args = parser.parse_args()
    main(args)
