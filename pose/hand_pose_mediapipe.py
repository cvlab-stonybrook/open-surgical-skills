# This code file detect hand landmarks and hand gesture with hand tracking strategies

import os
import argparse
import cv2
import glob
import pickle
import urllib.request
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, required=True, help='dataset root containing the Data and Analysis folders')
parser.add_argument('--model_path', type=str, default=None, help='path to the mediapipe gesture_recognizer.task model file. If not set, it is downloaded next to this script')
parser.add_argument('--group', type=str, default='Students', help='students or residents or doctors')
parser.add_argument("--subj_folder", type=str, required=True, help='folder for subject containing images for different cameras')
parser.add_argument("--camera", type=str, default='')
parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.JPG', '.png'])
parser.add_argument('--vis', action='store_true')
parser.add_argument('--fps', type=int, default= 30, help='the fps for loading the video for processing')
args = parser.parse_args()

subj_dir = os.path.join(args.base_dir, 'Data', args.group, args.subj_folder, "Images")
result_dir = os.path.join(args.base_dir, 'Data', args.group, args.subj_folder, 'handpose')
os.makedirs(result_dir, exist_ok=True)
output_dir = os.path.join(result_dir)


MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
model_path = args.model_path
if model_path is None:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gesture_recognizer.task')
    if not os.path.exists(model_path):
        print(f'Downloading the mediapipe model to {model_path}')
        urllib.request.urlretrieve(MODEL_URL, model_path)
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2)
        
#options = HandLandmarkerOptions(
        #base_options=BaseOptions(model_asset_path=model_path),
        #running_mode=VisionRunningMode.VIDEO,
        #num_hands=2)

real_hand_mapping = {0:'Right', 1:'Left'}  # I found the mediapipe always predict the opposite hand for handedness, so I defined the opposite mapping

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
GESTURE_TEXT_COLOR = (88, 205, 54) # vibrant green
if args.camera:
        all_camdirs = [os.path.join(subj_dir, args.camera)]
else:
        all_camdirs = sorted(glob.glob(os.path.join(subj_dir, 'Cam*')))

for cam_dir in all_camdirs:
        cam_name = cam_dir.split('/')[-1] 
        image_files = sorted(glob.glob(os.path.join(cam_dir, "*" + args.ext)))	
        save_dct = {}
        output_dir = os.path.join(result_dir, cam_name)
        if args.vis:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        with GestureRecognizer.create_from_options(options) as recognizer:
            #with HandLandmarker.create_from_options(options) as landmarker:
            # The detector is initialized. Use it here.
            # ...
                for img_idx, img_file in enumerate(tqdm(image_files)):
                    imgname = img_file.split('/')[-1]
                    # replace with .jpg
                    imgname = imgname.replace(args.ext, '.jpg')
                    
                    #real_idx = img_idx * args.fps
                    #frame_timestamp_ms = round(real_idx / args.fps_ori * 1000)
                    frame_timestamp_ms = round(img_idx / args.fps * 1000)
                    
                    mp_image = mp.Image.create_from_file(img_file)
                    height, width = mp_image.height, mp_image.width
                    #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

                    #hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                    results = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)        # gesture recognition result
                    hand_info = []        
                    image = mp_image.numpy_view()[:,:,::-1]
                    annotated_image = image.copy()

                    
                    for idx, hand_landmarks in enumerate(results.hand_landmarks):
                        # hand_landmarks is the landmarks detected for 1 hand
                            this_landmark, x_coordinates, y_coordinates = [], [], []
                            for landmark in hand_landmarks:
                                    this_landmark.append([landmark.x * width, landmark.y * height])
                                    y_coordinates.append(landmark.y)
                                    x_coordinates.append(landmark.x)
                                    
                            this_landmark = np.array(this_landmark)
                            handness = real_hand_mapping[results.handedness[idx][0].index]
                            gesture = results.gestures[idx][0].category_name           
                            #print ("handness: {}, keypoints: {}".format(handness, this_landmark))
                            hand_info.append({'handness': handness, 'keypoints':this_landmark})             
                    # handle cases when both hands are detected but with the same handness
                    
                    if len(hand_info) == 0:
                        print(f"No hands detected for {imgname}")
                    
                    if len(hand_info)==2:
                        hands_unique = set([each['handness'] for each in hand_info])
                        x1, x2 = np.mean(hand_info[0]['keypoints'][:,0]), np.mean(hand_info[1]['keypoints'][:,0])
                        y1, y2 = np.mean(hand_info[0]['keypoints'][:,1]), np.mean(hand_info[1]['keypoints'][:,1])
                        ori_hand1, ori_hand2 = hand_info[0]['handness'], hand_info[1]['handness']
                        if len(hands_unique)==1:
                            # New logic: use temporal consistency to resolve handedness
                            prev_left_kps = None
                            prev_right_kps = None
                            prev_left_kps_idx = None
                            prev_right_kps_idx = None
                            # Search backwards through previous frames for the last known positions of left and right hands
                            for look_back in range(1, img_idx + 1):
                                prev_idx = img_idx - look_back
                                if prev_idx < 0:
                                    break
                                prev_imgname = os.path.basename(image_files[prev_idx])
                                prev_imgname = prev_imgname.replace(args.ext, '.jpg')
                                prev_info = save_dct.get(prev_imgname, [])
                                
                                if prev_info:
                                    prev_map = {d['handness']: d['keypoints'] for d in prev_info if 'handness' in d and 'keypoints' in d}
                                    if prev_left_kps is None and 'Left' in prev_map:
                                        prev_left_kps = prev_map['Left']
                                        prev_left_kps_idx = int(image_files[prev_idx].split('/')[-1].split('.')[0])
                                    if prev_right_kps is None and 'Right' in prev_map:
                                        prev_right_kps = prev_map['Right']
                                        prev_right_kps_idx = int(image_files[prev_idx].split('/')[-1].split('.')[0])
                                if prev_left_kps is not None and prev_right_kps is not None:
                                    break
                            kps1 = hand_info[0]['keypoints']
                            kps2 = hand_info[1]['keypoints']
                            
                 
                            if prev_left_kps is not None and prev_right_kps is not None:
                                dist1_left = np.mean(np.linalg.norm(kps1 - prev_left_kps, axis=1))
                                dist2_left = np.mean(np.linalg.norm(kps2 - prev_left_kps, axis=1))
                                dist1_right = np.mean(np.linalg.norm(kps1 - prev_right_kps, axis=1))
                                dist2_right = np.mean(np.linalg.norm(kps2 - prev_right_kps, axis=1))
                                
                                # Assign based on minimizing sum of distances to previous known positions
                                if dist1_left + dist2_right < dist1_right + dist2_left:
                                    hand_info[0]['handness'] = 'Left'
                                    hand_info[1]['handness'] = 'Right'
                                else:
                                    hand_info[0]['handness'] = 'Right'
                                    hand_info[1]['handness'] = 'Left'
                            elif prev_left_kps is not None or prev_right_kps is not None:
                                if prev_left_kps is not None:
                                    dist1_left = np.mean(np.linalg.norm(kps1 - prev_left_kps, axis=1))
                                    dist2_left = np.mean(np.linalg.norm(kps2 - prev_left_kps, axis=1))
                                    if dist1_left < dist2_left:
                                        hand_info[0]['handness'] = 'Left'
                                        hand_info[1]['handness'] = 'Right'
                                    else:
                                        hand_info[0]['handness'] = 'Right'
                                        hand_info[1]['handness'] = 'Left'
                                elif prev_right_kps is not None:
                                    dist1_right = np.mean(np.linalg.norm(kps1 - prev_right_kps, axis=1))
                                    dist2_right = np.mean(np.linalg.norm(kps2 - prev_right_kps, axis=1))
                                    if dist1_right < dist2_right:
                                        hand_info[0]['handness'] = 'Right'
                                        hand_info[1]['handness'] = 'Left'
                                    else:
                                        hand_info[0]['handness'] = 'Left'
                                        hand_info[1]['handness'] = 'Right'
                            else:
                                # Fallback to spatial heuristics if no previous data is available
                                hand_1st, hand_2nd = ('Left', 'Right') if x1 > x2 else ('Right', 'Left')
                                hand_info[0]['handness'] = hand_1st
                                hand_info[1]['handness'] = hand_2nd
                            
                            
                            # if 'cam1' not in cam_name.lower(): 
                            #     if abs(x1-x2)>=100 or abs(y1-y2)>=100:
                            #         if cam_name=='Cam4':
                            #                 hand_1st, hand_2nd = ('Left', 'Right') if y1 < y2 else ('Right', 'Left')
                            #                 hand_info[0]['handness']= hand_1st
                            #                 hand_info[1]['handness']= hand_2nd
                            #         elif cam_name=='Cam3':
                            #             hand_1st, hand_2nd = ('Left', 'Right') if y1 > y2 else ('Right', 'Left')
                            #             hand_info[0]['handness']= hand_1st
                            #             hand_info[1]['handness']= hand_2nd
                            #         else:
                            #             hand_1st, hand_2nd = ('Left', 'Right') if x1 > x2 else ('Right', 'Left')
                            #             hand_info[0]['handness']= hand_1st
                            #             hand_info[1]['handness']= hand_2nd
                            #     else:
                            #         if 'cam4' in cam_name.lower():
                            #             hand_1st, hand_2nd = ('Left', 'Right') if x1 < x2 else ('Right', 'Left')
                            #             hand_info[0]['handness']= hand_1st
                            #             hand_info[1]['handness']= hand_2nd
                            # elif cam_name=='Cam1' and abs(x1-x2)>=20 or abs(y1-y2)>=20:
                            #     hand_1st, hand_2nd = ('Left', 'Right') if x1 > x2 else ('Right', 'Left')
                            #     hand_info[0]['handness']= hand_1st
                            #     hand_info[1]['handness']= hand_2nd
                            print(f"Switching handness for {imgname} from {ori_hand1} to {hand_info[0]['handness']} and {ori_hand2} to {hand_info[1]['handness']}")
                            print(f"x1: {np.mean(hand_info[0]['keypoints'][:,0])}, x2: {np.mean(hand_info[1]['keypoints'][:,0])}, y1: {np.mean(hand_info[0]['keypoints'][:,1])}, y2: {np.mean(hand_info[1]['keypoints'][:,1])}")

                    # elif len(hand_info)==1:
                    # 		#Handle single hand detection with potentially wrong handness label
                    # 	current_hand = hand_info[0]
                    # 	current_handness = current_hand['handness']
                    # 	current_keypoints = current_hand['keypoints']
                        
                    # 	# Search backwards through previous frames to find reference keypoints
                    # 	prev_hands_found = {}
                    # 	for look_back in range(1, min(img_idx + 1, 2)):  # Look back up to 2 frames
                    # 		prev_idx = img_idx - look_back
                    # 		prev_imgname = os.path.basename(image_files[prev_idx])
                    # 		prev_imgname = prev_imgname.replace(args.ext, '.jpg')
                    # 		prev_info = save_dct.get(prev_imgname, [])
                            
                    # 		if prev_info:
                    # 			for prev_hand in prev_info:
                    # 				if 'handness' in prev_hand and 'keypoints' in prev_hand:
                    # 					prev_handness = prev_hand['handness']
                    # 					if prev_handness not in prev_hands_found:
                    # 						prev_hands_found[prev_handness] = prev_hand['keypoints']
                                
                    # 			# If we found both handnesses, we can compare
                    # 			if len(prev_hands_found) >= 1:
                    # 				break
                        
                    # 	# Check if current handness exists in previous frames
                    # 	if current_handness not in prev_hands_found and len(prev_hands_found) > 0:
                    # 		# Current handness not found in previous frames, check distances to other handnesses
                    # 		best_match_handness = None
                    # 		best_distance = float('inf')
                            
                    # 		for prev_handness, prev_keypoints in prev_hands_found.items():
                    # 			# Calculate distance between current keypoints and previous keypoints
                    # 			dist = np.mean(np.sqrt(np.sum((current_keypoints - prev_keypoints) ** 2, axis=1)))
                                
                    # 			if dist < best_distance:
                    # 				best_distance = dist
                    # 				best_match_handness = prev_handness
                            
                    # 		# If the best match is reasonably close, correct the handness
                    # 		distance_threshold = min(width, height) * 0.2  # 20% of image dimension
                    # 		if best_match_handness and best_distance < distance_threshold:
                    # 			original_handness = current_handness
                    # 			hand_info[0]['handness'] = best_match_handness
                    # 			print(f"Correcting single hand handness for {imgname} from {original_handness} to {best_match_handness} (distance: {best_distance:.2f})")
                    
                    # Filter out hands with overly sparse keypoints (too spread out)
                    
                    filtered_hand_info = []
                    for i, hand_data in enumerate(hand_info):
                        keypoints = hand_data['keypoints']
                        # Calculate centroid (average point) of the hand pose
                        centroid = np.mean(keypoints, axis=0)
                        
                        # Calculate average distance from each keypoint to the centroid
                        distances_to_centroid = np.sqrt(np.sum((keypoints - centroid) ** 2, axis=1))
                        avg_distance_to_centroid = np.mean(distances_to_centroid)
                        
                        # Threshold for sparse keypoints (adjust based on image resolution and typical hand size)
                        if cam_name.lower().startswith('cam1'):
                                sparse_threshold = min(width, height) * 0.1  # 8% of the smaller image dimension
                        elif cam_name.lower().startswith('cam2'):
                                sparse_threshold = min(width, height) * 0.5  # 5% of the smaller image dimension
                        else:
                                sparse_threshold = min(width, height) * 0.3  # 8% of the smaller image dimension
                        if avg_distance_to_centroid > sparse_threshold:
                            print(f"Discarding {hand_data['handness']} hand for {imgname} due to sparse keypoints (avg dist to centroid: {avg_distance_to_centroid:.2f} > {sparse_threshold:.2f})")
                        else:
                            filtered_hand_info.append(hand_data)
                    
                    hand_info = filtered_hand_info

                    # Filter out hands with a much smaller scale compared to the previous frame
                    filtered_hand_info_scale = []
                    
                    
                    for hand_data in hand_info:
                        handness = hand_data['handness']
                        current_keypoints = hand_data['keypoints']

                        # Calculate current scale
                        current_centroid = np.mean(current_keypoints, axis=0)
                        current_distances = np.sqrt(np.sum((current_keypoints - current_centroid) ** 2, axis=1))
                        current_scale = np.mean(current_distances)

                        prev_scale = None
                        # Search backwards for the same handness
                        for look_back in range(1, min(img_idx + 1, 8)): # Look back up to 5 frames for efficiency
                            prev_idx = img_idx - look_back
                            prev_imgname = os.path.basename(image_files[prev_idx])
                            prev_imgname = prev_imgname.replace(args.ext, '.jpg')
                            prev_info = save_dct.get(prev_imgname, [])
                            
                            if prev_info:
                                prev_map = {d['handness']: d['keypoints'] for d in prev_info if 'handness' in d and 'keypoints' in d}
                                if handness in prev_map:
                                    prev_keypoints = prev_map[handness]
                                    # Calculate previous scale
                                    prev_centroid = np.mean(prev_keypoints, axis=0)
                                    prev_distances = np.sqrt(np.sum((prev_keypoints - prev_centroid) ** 2, axis=1))
                                    prev_scale = np.mean(prev_distances)
                                    break # Found it
                        
                        if prev_scale is not None and prev_scale > 1e-6: # Avoid division by zero
                            scale_ratio = current_scale / prev_scale
                            # Threshold for scale change, adjust if necessary
                            scale_threshold = 0.4 
                            #print(f"imgname: {imgname}, scale_ratio: {scale_ratio}, current_scale: {current_scale}, prev_scale: {prev_scale}")
                            if scale_ratio < scale_threshold: 
                                print(f"Discarding {handness} hand for {imgname} due to small scale compared to previous frame (current: {current_scale:.2f}, prev: {prev_scale:.2f}, ratio: {scale_ratio:.2f})")
                                continue # Skip adding this hand
                        
                        filtered_hand_info_scale.append(hand_data)
                    
                    hand_info = filtered_hand_info_scale

                    # Filter out falsely classified hands 
                    if len(hand_info) == 2 and 'cam4' not in cam_name.lower():
                        # Check if 80% of corresponding keypoints are within 10 pixels
                        keypoints1 = hand_info[0]['keypoints']
                        keypoints2 = hand_info[1]['keypoints']
                        
                        # Calculate distances between corresponding keypoints
                        distances = np.sqrt(np.sum((keypoints1 - keypoints2) ** 2, axis=1))
                        if cam_name.lower().startswith('cam1'):
                            thres = 20 
                        elif cam_name.lower().startswith('cam2'):
                            thres = 200
                        else:
                            thres = 100
                        close_keypoints = np.sum(distances < thres)
                        total_keypoints = len(keypoints1)
                        #print(f"image: {imgname}, close_keypoints: {close_keypoints}, total_keypoints: {total_keypoints}")
                        
                        if close_keypoints / total_keypoints > 0.6:
                            # Hands are too similar, find which one to discard based on previous frames
                            distances_to_prev = []
                            
                            for hand_data in hand_info:
                                handness = hand_data['handness']
                                current_keypoints = hand_data['keypoints']
                                best_dist = float('inf')
                                
                                # Search backwards through previous frames to find the same handness
                                for look_back in range(1, min(img_idx + 1, 20)):  # Look back up to 20 frames
                                    prev_idx = img_idx - look_back
                                    prev_imgname = os.path.basename(image_files[prev_idx])
                                    prev_imgname = prev_imgname.replace(args.ext, '.jpg')
                                    prev_info = save_dct.get(prev_imgname, [])
                                    
                                    if prev_info:
                                        prev_map = {d['handness']: d['keypoints'] for d in prev_info if 'handness' in d and 'keypoints' in d}
                                        if handness in prev_map:
                                            prev_keypoints = prev_map[handness]
                                            dist = np.mean(np.sqrt(np.sum((current_keypoints - prev_keypoints) ** 2, axis=1)))
                                            best_dist = dist
                                            break  # Found the most recent frame with this handness
                                
                                distances_to_prev.append(best_dist)
                            
                            # Keep the hand with smaller distance to previous frame
                            if len(distances_to_prev) == 2:
                                if abs(distances_to_prev[0] - distances_to_prev[1]) > 10:
                                    keep_idx = 0 if distances_to_prev[0] < distances_to_prev[1] else 1
                                    discard_idx = 1 - keep_idx
                                    
                                    print(f"Discarding falsely classified hand for {imgname} (distance: {distances_to_prev[discard_idx]:.2f} vs {distances_to_prev[keep_idx]:.2f})")
                                    print(f"hand kept: {hand_info[keep_idx]['handness']}")	
                                    hand_info = [hand_info[keep_idx]]
  
                    # Now visualize all hands (including backfilled ones)
                    if args.vis:
                        for hand_data in hand_info:
                            # Convert keypoints back to normalized coordinates for visualization
                            keypoints = hand_data['keypoints']
                            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                            hand_landmarks_proto.landmark.extend([
                                landmark_pb2.NormalizedLandmark(x=float(pt[0])/width, y=float(pt[1])/height, z=0.0) 
                                for pt in keypoints
                            ])
                            
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                hand_landmarks_proto,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                    if args.vis:
                        cv2.imwrite(os.path.join(output_dir, imgname), annotated_image)
                        
                    save_dct[imgname] = hand_info                
                    
        with open(os.path.join(result_dir, f'handpose_{cam_name}.pkl'), 'wb') as file:
            pickle.dump(save_dct, file) 
