#!/bin/bash
# Run a per-subject command for every subject folder of a group.
# Subject folders are the ones whose name starts with a number.
# Usage: bash scripts/run_group.sh <data_root> <group> <command...>
#   {subj} in the command is replaced by the subject folder name.
# Example:
#   bash scripts/run_group.sh $DATA_ROOT Attendings python pose/hand_pose_mediapipe.py \
#       --base_dir $DATA_ROOT --group Attendings --subj_folder {subj} --fps 5
if [ "$#" -lt 3 ]; then
	echo "Usage: $0 <data_root> <group> <command...>"
	exit 1
fi
group_dir="$1/Data/$2"
shift 2
if [ ! -d "$group_dir" ]; then
	echo "Error: group directory '$group_dir' does not exist"
	exit 1
fi
for subj_dir in "$group_dir"/*; do
	subj=$(basename "$subj_dir")
	if [ -d "$subj_dir" ] && [[ "$subj" =~ ^[0-9] ]]; then
		echo "Processing subject: $subj"
		"${@//\{subj\}/$subj}" || echo "Error processing $subj"
	fi
done
