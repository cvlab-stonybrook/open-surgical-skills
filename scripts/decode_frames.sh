#!/bin/bash
# Decode the videos of all views of a subject into frames at 5 fps (every 6th frame of the 30 fps videos)
# Usage: bash scripts/decode_frames.sh <data_root> <group> <subj_folder>
if [ "$#" -ne 3 ]; then
	echo "Usage: $0 <data_root> <group> <subj_folder>"
	exit 1
fi
subj_dir="$1/Data/$2/$3"
for cam in Cam1 Cam2 Cam3 Cam4 Ego; do
	if [ -f "$subj_dir/Videos/${cam}.mp4" ]; then
		mkdir -p "$subj_dir/Images/${cam}"
		ffmpeg -v error -i "$subj_dir/Videos/${cam}.mp4" -vf fps=5 -start_number 0 "$subj_dir/Images/${cam}/%06d.jpg"
	fi
done
