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


folder_path = r"C:\Users\keela\Coding\Models\FinalResults\MLE\initial_detections.pkl"
output_path = r"C:\Users\keela\Coding\Models\FinalResults\MLE\sort_results.pkl"

# folder_path = r"initial_detections_2L.pkl"
# output_path = r"sort_results_2L.pkl"


def bb_intersection_over_union(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou

def match_boxes(boxes, tracks):
    assignments = []
    for track in tracks:
        track_ious = []
        for box in boxes:
            boxa = track[0:4]
            boxb = box
            track_ious.append(bb_intersection_over_union(boxa, boxb))
        track_association = np.argmax(track_ious)
        assignments.append(track_association)

    return assignments

# Load data from a pickle file
with open(folder_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)

# Initialize SORT tracker
mot_tracker = sort.Sort(min_hits=1, iou_threshold=.05, max_age=3)

# Create and open a pickle file for writing tracking results
with open(output_path, 'wb') as pickle_file:
    results_dict = {}

    for frame_num, frame_data in loaded_frames_detections.items():
        boxes = frame_data['boxes']
        probabilities = frame_data['cls_prob']

        # Use SORT to update object tracking
        try:
            track_bbs_ids = mot_tracker.update(boxes)
            assignments = match_boxes(boxes, track_bbs_ids)
            assigned_probs = [probabilities[a] for a in assignments]
            results_dict[frame_num] = {'boxes': track_bbs_ids[:,0:4], 'cls_prob': assigned_probs, 'track_id':track_bbs_ids[:,-1]}
        except:
            print(f"Nothing in frame {frame_num}")
            results_dict[frame_num] = {'boxes': np.array(boxes), 'cls_prob': np.array(probabilities), 'track_id':np.array([[]])}


    # Save the results dictionary to the pickle file
    pickle.dump(results_dict, pickle_file)
