import os
import sort
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import random
from motmetrics import metrics, utils  # Import motmetrics modules
import csv

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


# Get object class dict
with open(r'D:\Coding\Thesis\sort\object_dict.txt', 'r') as file:
    color_mapping = file.read()
    
colors = generate_random_colors(len(color_mapping))  # Generate random colors based on the number of classes

# Initialize SORT tracker
mot_tracker = sort.Sort()

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Path to the image folder
image_folder = Path(r"D:\Coding\Thesis\MOT17\train\MOT17-13-DPM\img1")

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
with open('detect_and_sort_results.csv', mode='w', newline='') as csv_file:
    fieldnames = ['FrameNumber', 'ObjectID', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for image_path in image_folder.iterdir():
        image = cv2.imread(str(image_path))
        results = model.predict(image)[0]
        
        detections = []
        boxes = results.boxes
        confs = boxes.data[:, 4:6]
        classes = []
        for box, conf in zip(boxes, confs):
            b = box.xyxy[0].tolist()
            c = conf.tolist()
            b.append(c[0])
            detections.append(b)
            classes.append(c[1])
        track_bbs_ids = mot_tracker.update(detections)
        print(track_bbs_ids)

        for (xmin, ymin, xmax, ymax, obj_id), class_index in zip(track_bbs_ids, classes):
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            class_name = f"Class {color_mapping[int(class_index)]}"  # Cast class_index to int for class name
            object_id_and_class = f"Object ID: {int(obj_id)}, {class_name}"  # Concatenate Object ID and Class
            color = colors[int(class_index)] if class_index < len(colors) else (0, 0, 0)  # Get a random color for the class
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, object_id_and_class, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
            # Write the tracking result to the CSV file
            writer.writerow({
                'FrameNumber': image_path.stem,  # Assuming frame numbers are the image file names without extension
                'ObjectID': int(class_index),  # Cast class_index to int for ObjectID
                'X': xmin,
                'Y': ymin,
                'Width': xmax - xmin,
                'Height': ymax - ymin,
                'Confidence': c[0],  # Use the confidence from YOLO results
                'Class': class_name
            })

        # Write the frame with bounding boxes to the output video
        out.write(image)

# Release the VideoWriter object and close the video file
out.release()


print(f'Video saved as {output_video_path}')















'''
from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\PathToYourModel\model.pt")

# Predict a class of an image
results = model(r"C:\PathToYourImage\image.jpg")

# Print the result
if results[0].probs[0] > results[0].probs[1]:
    print("First class")
else:
    print("Second class")
'''