import sys
sys.path.append('/remote_home/Thesis')

import argparse
import tensorflow as tf
import numpy as np
from utils.coco_dataset_manager import *
import xml.etree.ElementTree as ET
import tensorflow as tf
import keras_cv
from utils.yolo_utils import *
from utils.custom_retinanet import prepare_image
from utils.nonmaxsuppression import PreBayesianNMS
from pycocotools.coco import COCO
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os
import ast
import json
import pickle
import random
import sort
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import random
#import matplotlib
#matplotlib.use('Agg')


# Define the path for the parsed dictionary of objects found in frame
dict_path = '/remote_home/Thesis/Sort/parsed_data_dict.pkl'

# Define a function to generate random colors
def generate_random_colors(num_colors):
    random.seed(42)  # Set a seed for reproducibility
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((r, g, b))
    return colors

# Load data from a pickle file
with open(dict_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)

colors = generate_random_colors(len(loaded_frames_detections))  # Generate random colors based on the number of frames

# Initialize SORT tracker
mot_tracker = sort.Sort()

# Path to the image folder
image_folder = Path(r"/remote_home/Thesis/BDD_Files/traffic")

# Get the dimensions of the first image in the folder
first_image = next(image_folder.iterdir())
first_image = cv2.imread(str(first_image))
height, width, _ = first_image.shape

# Define the output video path
output_video_path = 'output_video_detect_sort.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# Create and open a CSV file for writing tracking results
with open('detect_and_sort_results_2.csv', mode='w', newline='') as csv_file:
    fieldnames = ['FrameNumber', 'ObjectID', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Sort image file paths based on filenames
    image_paths = sorted(image_folder.iterdir(), key=lambda x: int(os.path.splitext(x.name)[0]))

    for frame_num, frame_data in loaded_frames_detections.items():
        image_path = next((p for p in image_paths if frame_num == os.path.splitext(p.name)[0]), None)

        if image_path is not None:
            image = cv2.imread(str(image_path))
            boxes = frame_data['boxes']
            probabilities = frame_data['probabilities']
            classes = frame_data['classes']

            detections = []
            for box, confidence, class_index in zip(boxes, probabilities, classes):
                b = box
                c = confidence
                b.append(c)
                detections.append(b)

            # Use SORT to update object tracking
            track_bbs_ids = mot_tracker.update(detections)

            for (xmin, ymin, xmax, ymax, obj_id), confidence, class_index in zip(track_bbs_ids, probabilities, classes):
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                class_name = f"Class {class_index}"  # Adjust this part according to your class mapping

                object_id_and_class = f"{class_name}: {confidence:.3f}"
                color = colors[int(frame_num)] if int(frame_num) < len(colors) else (0, 0, 0)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, object_id_and_class, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Write the tracking result to the CSV file
                writer.writerow({
                    'FrameNumber': frame_num,
                    'ObjectID': int(obj_id),
                    'X': xmin,
                    'Y': ymin,
                    'Width': xmax - xmin,
                    'Height': ymax - ymin,
                    'Confidence': float(confidence),
                    'Class': class_index
                })

            # Write the frame with bounding boxes to the output video
            out.write(image)

# Release the VideoWriter object and close the video file
out.release()

print(f'Video saved as {output_video_path}')
