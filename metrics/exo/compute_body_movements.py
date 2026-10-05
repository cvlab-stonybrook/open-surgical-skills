# count the large body movements (spine / neck) from the MotionBERT 3d body pose for all subjects,
# and write them to Analysis/body_movements.xlsx, which is read by aggregate_and_correlate.py
import os
import argparse
import numpy as np
import pandas as pd

# Human3.6M keypoint indices used by MotionBERT
PART_KEYPOINT = {'Spine': 7, 'Neck': 8}


def pixel2world_vis_motion(motion):
    # follow the visualization code in MotionBert to convert to world coordinate
    # motion: (T,17,3)
    motion = np.transpose(motion, (1, 2, 0))   # (T,17,3) -> (17,3,T)
    offset = np.ones([3, motion.shape[-1]]).astype(np.float32)
    offset[2, :] = 0
    new_motion = (motion + offset) * 512 / 2
    new_motion = new_motion.transpose((2, 0, 1))
    new_motion[:, :, [1, 2]] = new_motion[:, :, [2, 1]]
    new_motion = -1 * new_motion
    return new_motion


def count_large_movements(kp_3d_all, img_list, kp_idx, fps, velo_thres, start_frame=0, end_frame=-1):
    """Count the movements where the keypoint velocity exceeds the threshold for at least 2 consecutive frames."""
    if end_frame == -1:
        end_frame = int(os.path.splitext(img_list[-1])[0])
    time_interv = 1 / fps

    num_exceed = 0
    kp_prev = None
    last_move = False  # whether last frame has large movement
    counted = False    # only count one time for each consecutive movement
    for idx, imgname in enumerate(img_list):
        img_idx = int(os.path.splitext(imgname)[0])
        if img_idx < start_frame:
            continue
        if img_idx > end_frame or idx >= len(kp_3d_all):
            break
        kp = kp_3d_all[idx][kp_idx]
        if kp_prev is not None:
            velo = np.linalg.norm((kp - kp_prev) / time_interv)
            if velo >= velo_thres:
                if last_move and not counted:
                    num_exceed += 1
                    counted = True
            else:
                counted = False
            last_move = velo >= velo_thres
        kp_prev = kp
    return num_exceed


def main(args):
    data_dir = os.path.join(args.base_dir, 'Data')
    results = {}  # sheet name -> list of (subject, count)
    for group in args.groups:
        group_dir = os.path.join(data_dir, group)
        # subject folders start with a number (skips e.g. Calibration_combined, not_used)
        subj_folders = sorted(f for f in os.listdir(group_dir)
                              if f[0].isdigit() and os.path.isdir(os.path.join(group_dir, f)))
        for subj_folder in subj_folders:
            subj_dir = os.path.join(group_dir, subj_folder)
            body_3d_path = os.path.join(subj_dir, 'bodypose', 'X3D.npy')
            if not os.path.exists(body_3d_path):
                print(f"Warning: {body_3d_path} not found, skipping {group}/{subj_folder}")
                continue
            kp_3d_all = pixel2world_vis_motion(np.load(body_3d_path))
            img_list = sorted(os.listdir(os.path.join(subj_dir, 'Images', args.camera)))
            for part, kp_idx in PART_KEYPOINT.items():
                num = count_large_movements(kp_3d_all, img_list, kp_idx, args.fps, args.velo_thres)
                results.setdefault(f'{group}_{part}', []).append((subj_folder, num))
            print(f"{group}/{subj_folder}: " + ", ".join(f"{part} {results[f'{group}_{part}'][-1][1]}" for part in PART_KEYPOINT))

    output_path = os.path.join(args.base_dir, 'Analysis', args.output_file)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for part in ['Neck', 'Spine']:
            for group in args.groups:
                sheet_name = f'{group}_{part}'
                if sheet_name in results:
                    pd.DataFrame(results[sheet_name]).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help='Base directory containing the Data and Analysis folders')
    parser.add_argument('--groups', type=str, nargs='+', default=['Students', 'Residents', 'Attendings'])
    parser.add_argument('--camera', type=str, default='Cam1', help='the body-view camera used for body pose estimation')
    parser.add_argument('--fps', type=float, default=5, help='the fps of the decoded frames')
    parser.add_argument('--velo_thres', type=float, default=10.0)
    parser.add_argument('--output_file', type=str, default='body_movements.xlsx')
    args = parser.parse_args()
    main(args)
