# AlphaPose sometimes contain redundant detections, here I just keep the 1st detection for each image
import os
import numpy as np
import pickle
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, required=True, help='dataset root containing the Data and Analysis folders')
parser.add_argument('--group', type=str, default='Students', help='students or residents or doctors')
parser.add_argument("--subj_folder", type=str, required=True, help='folder for subject containing images for different cameras')
parser.add_argument("--ego", action='store_true', help='if specified, do for the ego folder, otherwise do all fixed camera folders')
args = parser.parse_args()


subj_dir = os.path.join(args.base_dir, 'Data', args.group, args.subj_folder)
ori_annt_path = os.path.join(subj_dir, "bodypose", "alphapose-results.json")


with open(ori_annt_path, 'r') as file:
    ori_annt = json.load(file)

img_set = set() 
new_annts = []

for annt in ori_annt:
    fname = annt['image_id'] 
    if fname in img_set:
        continue
    new_annts.append(annt)
    img_set.add(fname)
    
with open(os.path.join(subj_dir, "bodypose", "alphapose-results-cleaned.json"), 'w') as file: 
    json.dump(new_annts, file)

print(len(new_annts))
    
