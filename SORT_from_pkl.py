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

# Define the path for the parsed dictionary of objects found in frame



def SORT_initial_detections(dict_path = r"C:\Users\keela\Documents\Basic_BCE\initial_detections.pkl",\
                            output_path = r"C:\Users\keela\Documents\Basic_BCE\sort_results.pkl"):
    # Load data from a pickle file
    with open(dict_path, 'rb') as pickle_file:
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

            frame_results = []
            for (xmin, ymin, xmax, ymax, obj_id), confidence in zip(track_bbs_ids, probabilities):
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                frame_results.append({
                    'ObjectID': int(obj_id),
                    'X': xmin,
                    'Y': ymin,
                    'Width': xmax - xmin,
                    'Height': ymax - ymin,
                    'Confidence': float(np.max(confidence)),
                    'Class': np.argmax(confidence)
                })

            # Save the frame results to the dictionary
            results_dict[frame_num] = frame_results

        # Save the results dictionary to the pickle file
        pickle.dump(results_dict, pickle_file)

if __name__ == "__main":
    SORT_initial_detections()