# 2D pose extraction and 3D pose estimation (exocentric views)

This document describes how we obtain the 3D hand pose and the 3D body pose from the four fixed cameras. `Cam1` is the body-view camera, and `Cam2`, `Cam3`, `Cam4` are the hand-view cameras. The videos of the different views need to be synchronized, and the camera parameters (`intri.yml` and `extri.yml`, in the format of EasyMocap) are needed for the triangulation.

All of our scripts take `--base_dir`, the dataset root, which is organized as:

```
$DATA_ROOT
├── Analysis                      # annotations and computed metrics
└── Data
    └── {Group}                   # Students / Residents / Attendings
        └── {subj_folder}
            ├── Videos            # Cam1.mp4 ... Cam4.mp4, Ego.mp4
            ├── Images            # frames decoded at 5 fps: Cam1 ... Cam4, Ego
            ├── Calibration       # camera parameters: intri.yml, extri.yml
            ├── handpose          # 2D and 3D hand poses
            └── bodypose          # 2D and 3D body poses
```

## Installation

```
pip install -r requirements.txt
pip install -e third_party/EasyMocap
```

Our modified copy of [EasyMocap](https://github.com/zju3dv/EasyMocap), which we use for the multi-view triangulation, is included under `third_party/EasyMocap` (see `third_party/README.md` for its license and our changes). The following are installed separately, by following their own instructions:

- [CoTracker](https://github.com/facebookresearch/co-tracker) for recovering missed hand detections. We used commit `8d36403` with the `cotracker_stride_4_wind_8.pth` checkpoint.
- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for 2D body pose. We used the Halpe-26 ResNet-50 model (`halpe26_fast_res50_256x192.pth`).
- [MotionBERT](https://github.com/Walter0807/MotionBERT) for lifting the 2D body pose to 3D. We used the `FT_MB_lite_MB_ft_h36m_global_lite.bin` checkpoint.

## Frame decoding

All of our pose estimation is done at 5 fps, i.e. on every 6th frame of the 30 fps videos. Decode the videos of a subject into `Images/{Cam}` with:

`bash scripts/decode_frames.sh $DATA_ROOT {group} {subj_folder}`

## 3D hand pose

1. Detect the 2D hand poses in all views with MediaPipe:

   `python pose/hand_pose_mediapipe.py --base_dir $DATA_ROOT --group {group} --subj_folder {subj_folder} --fps 5` (add `--vis` to save visualizations). The MediaPipe model file is downloaded automatically on the first run.

   The results are saved in `handpose/handpose_{Cam}.pkl` as `dct[imgname] = [{'handness': 'Left' or 'Right', 'keypoints': 21 hand keypoints, 'gesture': gesture}, ...]`.

2. (Optional) Recover the hand poses that MediaPipe missed, by tracking the keypoints from neighboring frames with CoTracker. This requires the `cotracker` package to be installed (`pip install -e .` under the CoTracker folder) and a GPU:

   `python pose/filter_handpose_and_track.py --base_dir $DATA_ROOT --ckpt_path {path to cotracker_stride_4_wind_8.pth} --group {group} --subj_folder {subj_folder} --fps 5`

   The results are saved in `handpose/handpose_{Cam}_track.pkl`.

3. Triangulate the 2D hand poses to 3D. Under `third_party/EasyMocap`, run:

   `python apps/demo/hand_get3d.py $DATA_ROOT/Data/{group}/{subj_folder} --out $DATA_ROOT/Data/{group}/{subj_folder}/vis_3d --annotfolder handpose --imagefolder Images --sub Cam1 Cam2 Cam3 Cam4 --sub_vis Cam2 Cam3 Cam4 --annot_file handpose.pkl --cam_as_world Cam3`

   Use `--annot_file handpose_track.pkl` if the previous step was run. Add `--vis_repro` to visualize the 3D pose and its reprojections. The results are saved in `handpose/keypoints_3d.pickle`.

4. Smooth the 3D hand pose and interpolate the missing poses:

   `python triangulation/smooth_hand3d.py --base_dir $DATA_ROOT --groups {group} --subj_name {subj_folder} --annot_file handpose.pkl`

   The results are saved in `handpose/keypoints_3d_smoothed.pickle`. Add `--use_easymocap_vis --out_dir {folder name}` to visualize the reprojections of the smoothed pose.

To run a step for all subjects of a group, use `scripts/run_group.sh` (see the example inside the script).

## 3D body pose

1. Detect the 2D body pose from the body-view camera with AlphaPose. Under the AlphaPose folder, run:

   `python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint {path to halpe26_fast_res50_256x192.pth} --indir $DATA_ROOT/Data/{group}/{subj_folder}/Images/Cam1 --outdir $DATA_ROOT/Data/{group}/{subj_folder}/bodypose --detector yolo`

2. AlphaPose sometimes outputs multiple detections for the same image. Keep the first one:

   `python pose/alphapose/clean_detections.py --base_dir $DATA_ROOT --group {group} --subj_folder {subj_folder}`

3. Lift the 2D body pose to 3D with MotionBERT. Under the MotionBERT folder, run:

   `python infer_wild.py --json_path $DATA_ROOT/Data/{group}/{subj_folder}/bodypose/alphapose-results-cleaned.json --out_path $DATA_ROOT/Data/{group}/{subj_folder}/bodypose --evaluate {path to FT_MB_lite_MB_ft_h36m_global_lite.bin}`

   Add `-v pose3d` to output a video. The results are saved in `bodypose/X3D.npy`, in the [Human3.6M keypoint format](https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html).
