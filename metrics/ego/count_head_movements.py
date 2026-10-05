#!/usr/bin/env python3
# get the number of large head movements from the movement scores computed from the egocentric frames
import os
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

def main(args):
    """
    Loads head movement scores, segments them by surgical cycle, counts large
    movements per cycle, and saves the per-subject average to an Excel file.
    """
    analysis_dir = os.path.join(args.base_dir, 'Analysis')
    args.input_file = os.path.join(analysis_dir, 'head_movement_scores_ego.pkl')
    args.output_file = os.path.join(analysis_dir, 'head_movement_avg_per_cycle.xlsx')

    # --- Load Cycle Annotations ---
    excel_path = os.path.join(analysis_dir, 'cycle_annotations.xlsx')
    try:
        df_cycles = pd.read_excel(excel_path, header=0, index_col=0)
    except FileNotFoundError:
        print(f"ERROR: Cycle annotation file not found at {excel_path}")
        return

    # --- Load Head Movement Scores ---
    try:
        with open(args.input_file, 'rb') as f:
            head_movement_scores = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Head movement scores file not found at {args.input_file}")
        return

    results = {}
    for group in head_movement_scores.keys():
        results[group] = {}
        
        print(f"\nProcessing group: {group}")
        for subj, scores_data in tqdm(head_movement_scores[group].items(), desc=f"Subjects in {group}"):
            
            # --- Get Cycle Timings ---
            try:
                cycle_key = f"{group}_{subj}"
                cycles_subj = df_cycles.loc[cycle_key].dropna().values[::2]
                
                fps = args.fps
                
                cycle_times_sec = np.array([int(c.split(':')[0]) * 60 + int(c.split(':')[1]) for c in cycles_subj])
                cycle_frame_boundaries = cycle_times_sec * fps
            except KeyError:
                print(f"  - WARNING: No cycle annotations for {group}/{subj}. Skipping.")
                continue

            # --- Count Large Movements Within Each Cycle ---
            movements_per_cycle = []
            # n boundaries define n-1 cycles
            for i in range(len(cycle_frame_boundaries) - 1):
                start_frame = cycle_frame_boundaries[i]
                end_frame = cycle_frame_boundaries[i + 1]
                
                large_movements_in_cycle = 0
                if not scores_data: continue

                for item in scores_data:
                    frame_idx = item['frame_idx']
                    score = item['score']
                    if start_frame <= frame_idx < end_frame:
                        # score == -1 (insufficient matches) is also counted as a large movement
                        if score > args.movement_threshold or score == -1:
                            large_movements_in_cycle += 1
                
                movements_per_cycle.append(large_movements_in_cycle)
            
            # --- Calculate Average Per Cycle ---
            avg_movements = np.mean(movements_per_cycle) if movements_per_cycle else 0.0
            total_movements = np.sum(movements_per_cycle)
            results[group][subj] = {'Avg_Large_Head_Movements_Per_Cycle': avg_movements, 'Total_Large_Head_Movements': total_movements}

    # --- Save results to an Excel file ---
    with pd.ExcelWriter(args.output_file) as writer:
        for group, subj_data in results.items():
            if not subj_data: continue
            write_data = [(subj, subj_data[subj]['Avg_Large_Head_Movements_Per_Cycle'], subj_data[subj]['Total_Large_Head_Movements']) for subj in subj_data.keys()]
            df_out = pd.DataFrame(write_data, columns=['Subject', 'Avg_Large_Head_Movements_Per_Cycle', 'Total_Large_Head_Movements'])
            df_out.sort_values(by='Subject', inplace=True)
            df_out.to_excel(writer, sheet_name=group, index=False)
            
            # Print average for the group
            if not df_out.empty:
                group_avg = df_out['Avg_Large_Head_Movements_Per_Cycle'].mean()
                print(f"Average for {group}: {group_avg:.2f} large movements per cycle")

    print(f"\nAnalysis complete. Results saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze head movement scores per surgical cycle.")
    parser.add_argument("--base_dir", type=str, required=True, help='dataset root containing the Data and Analysis folders')
    parser.add_argument("--fps", type=int, default=5, help='frame rate of the frame indices in the image names (5 if the frames are decoded with scripts/decode_frames.sh)')
    parser.add_argument('--movement_threshold', type=float, default=0.03,
                        help='Movement score threshold to count as a large movement.')
    args = parser.parse_args()
    main(args)
