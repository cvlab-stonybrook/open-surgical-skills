# Behavior metrics

This document describes how the behavior metrics are computed from the estimated poses, for the exocentric views and for the egocentric view.

All scripts take `--base_dir`, the dataset root (see [pose_and_triangulation.md](pose_and_triangulation.md) for its layout). The scripts go through the subjects under `Data/Students`, `Data/Residents` and `Data/Attendings`. A subject is identified by `{Group}_{subj_folder}`.

`--fps` is the frame rate of the frame indices in the image names. It is 5 when the frames are decoded with `scripts/decode_frames.sh`.

## Annotations

The following files are expected under `$DATA_ROOT/Analysis`:

- `cycle_annotations.xlsx`: the start and end time (`mm:ss`) of each suturing cycle. One row per subject, indexed by `{Group}_{subj_folder}`, with the columns alternating between the start time and the end time of the cycles. The metrics are computed per cycle, and the start times are used as the cycle boundaries.
- `left_dominant_list.txt`: the subjects whose dominant hand is the left hand, one `{Group},{subj_folder}` per line. We define the dominant hand as the hand that pulls out the suture. The other subjects are treated as right-dominant.
- `scores.xlsx`: the expert ratings, with the columns `Index`, `Score1`, `Scores2`, `Scores_Avg`, `Group`, `Subject_name`.
- `completion_times.xlsx`: the task completion time in seconds. One sheet per group, without header, with the subject folder in the first column and the time in the second column.

## Exocentric views

The hand metrics are computed from the smoothed 3D hand pose (`handpose/keypoints_3d_smoothed.pickle`) and the body metric from the 3D body pose (`bodypose/X3D.npy`).

1. Hand travel distance per cycle. This also splits the 3D hand poses into cycles (`Analysis/keypoints_by_cycles_smoothed.pkl`), which is used by the next two steps:

   `python metrics/exo/compute_hand_distance.py --base_dir $DATA_ROOT`

2. Hand motion smoothness (SPARC):

   `python metrics/exo/compute_hand_smoothness.py --base_dir $DATA_ROOT`

3. Hand motion consistency across cycles (Procrustes analysis):

   `python metrics/exo/compute_hand_stability.py --base_dir $DATA_ROOT`

4. Number of large body movements (spine and neck):

   `python metrics/exo/compute_body_movements.py --base_dir $DATA_ROOT`

5. Aggregate all metrics into `Analysis/aggregated_metrics.xlsx`:

   `python metrics/exo/aggregate_and_correlate.py --base_dir $DATA_ROOT`

   Add `--vis` or `--separate-plots` to plot the metrics by group.

The hand consistency columns in the aggregated file are saved as `1 - Procrustes distance`, so a larger value means more consistent motion.

## Egocentric view

In the egocentric view, the hand metrics are computed from the 2D hand pose, and the head movement is computed from the camera motion between consecutive frames.

1. Detect the 2D hand pose in the egocentric frames, then interpolate the missing detections:

   `python pose/hand_pose_mediapipe.py --base_dir $DATA_ROOT --group {group} --subj_folder {subj_folder} --camera Ego --fps 5`

   `python pose/smooth_hand2d.py --base_dir $DATA_ROOT`

   The results are saved in `handpose/handpose_Ego_smoothed.pkl`.

2. Split the 2D hand poses into cycles (`Analysis/keypoints_by_cycles_ego.pkl`):

   `python metrics/ego/split_keypoints_by_cycle.py --base_dir $DATA_ROOT`

3. Hand motion smoothness and consistency:

   `python metrics/ego/compute_hand_smoothness_ego.py --base_dir $DATA_ROOT`

   `python metrics/ego/compute_hand_stability_ego.py --base_dir $DATA_ROOT`

4. Head movement. We match [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) keypoints between consecutive egocentric frames with [LightGlue](https://github.com/cvg/LightGlue) (install it by following its instructions), estimate a homography, and use the displacement of the image corners as the movement score:

   `python metrics/ego/compute_head_movement_ego.py --base_dir $DATA_ROOT`

   Then count the frames with a large movement in each cycle. We use a threshold of 0.03 on the movement score:

   `python metrics/ego/count_head_movements.py --base_dir $DATA_ROOT`

5. Aggregate all metrics into `Analysis/aggregated_metrics_ego.xlsx`:

   `python metrics/ego/aggregate_and_correlate_ego.py --base_dir $DATA_ROOT`

## Skill prediction

`python predict/predict_exo.py --base_dir $DATA_ROOT` and `python predict/predict_ego.py --base_dir $DATA_ROOT` train and evaluate the regression models on `Analysis/aggregated_metrics.xlsx` and `Analysis/aggregated_metrics_ego.xlsx` with repeated 4-fold cross-validation. The results are saved under `Analysis/Prediction` and `Analysis/Prediction_Ego`.
