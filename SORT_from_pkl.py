import sys
sys.path.append('/remote_home/Thesis')

import numpy as np
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import pickle
import sort
from pathlib import Path
import cv2
import numpy as np
import matplotlib

import csv
matplotlib.use('Agg')


folder_path = r"C:\Users\keela\Coding\Models\LongRuns\MLE_L2_Sigmoid\initial_detections.pkl"
output_path = r"C:\Users\keela\Coding\Models\LongRuns\MLE_L2_Sigmoid\sort_results.pkl"

# Load data from a pickle file
with open(folder_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)

# Initialize SORT tracker
mot_tracker = sort.Sort()

# Create and open a pickle file for writing tracking results
with open(output_path, 'wb') as pickle_file:
    results_dict = {}

    for frame_num, frame_data in loaded_frames_detections.items():
        boxes = frame_data['boxes']
        probabilities = frame_data['cls_prob']

        # Use SORT to update object tracking
        try:
            track_bbs_ids = mot_tracker.update(boxes)
            results_dict[frame_num] = {'boxes': track_bbs_ids[:,0:4], 'cls_prob': probabilities, 'track_id':track_bbs_ids[:,-1]}
        except:
            print(f"Nothing in frame {frame_num}")
            results_dict[frame_num] = {'boxes': np.array(boxes), 'cls_prob': np.array(probabilities), 'track_id':np.array([[]])}

    # Save the results dictionary to the pickle file
    pickle.dump(results_dict, pickle_file)
