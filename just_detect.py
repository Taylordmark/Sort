import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import random
import csv


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

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Path to the image folder
image_folder = Path(r"D:\Coding\Thesis\MOT17\train\MOT17-13-DPM\img1")

# Get the dimensions of the first image in the folder
first_image = next(image_folder.iterdir())
first_image = cv2.imread(str(first_image))
height, width, _ = first_image.shape

# Define the output video path
output_video_path = 'output_video_no_tracker.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))


# Create and open a CSV file for writing tracking results
with open('just_detect_results.csv', mode='w', newline='') as csv_file:
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
            xmin, ymin, xmax, ymax = map(int, b)
            class_index = int(c[1])
            confidence = c[0]
            
            class_name = f"Class {class_index}"  # Replace with your actual class names or labels
            color = colors[class_index] if class_index < len(colors) else (0, 0, 0)  # Get a random color for the class
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
            # Write the tracking result to the CSV file
            writer.writerow({
                'FrameNumber': image_path.stem,  # Assuming frame numbers are the image file names without extension
                'ObjectID': int(class_index),  # Cast class_index to int for ObjectID
                'X': xmin,
                'Y': ymin,
                'Width': xmax - xmin,
                'Height': ymax - ymin,
                'Confidence': confidence,  # Use the confidence from YOLO results
                'Class': class_name
            })

        # Write the frame with bounding boxes to the output video
        out.write(image)

# Release the VideoWriter object and close the video file
out.release()

print(f'Video saved as {output_video_path}')