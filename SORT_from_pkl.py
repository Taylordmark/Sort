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


folder_path = r"C:\Users\keela\Documents\Models\LastMinuteRuns\Small_MLE\000f8d37-d4c09a0f_initial_detections.pkl"
output_path = r"C:\Users\keela\Documents\Models\LastMinuteRuns\Small_MLE\sort_results.pkl"

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
        probabilities = frame_data['probabilities']

        detections = []
        for box, confidence in zip(boxes, probabilities):
            b = [box[0], box[1], box[0]+box[2], box[1]+box[3]]  # xywh to xyxy, as SORT wants xyxy format
            c = confidence
            detections.append(b)

        # Use SORT to update object tracking
        track_bbs_ids = mot_tracker.update(detections)

        results_dict[frame_num] = {'boxes': track_bbs_ids[:,0:4], 'probabilities': probabilities, 'track_id':track_bbs_ids[:,-1]}

    # Save the results dictionary to the pickle file
    pickle.dump(results_dict, pickle_file)