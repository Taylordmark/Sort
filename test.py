import os
import pickle
import cv2

# Load bounding boxes data from the pickle file
boxes_path = "/remote_home/Thesis/Sort/parsed_data_dict.pkl"
with open(boxes_path, 'rb') as file:
    bounding_boxes = pickle.load(file)

# Print the number of frames for verification
print(len(bounding_boxes))

# Path to the image folder
image_folder_path = '/remote_home/Thesis/BDD_Files/'

# Create a new folder for saving annotated images
output_folder_path = '/remote_home/Thesis/detections_with_boxes/'
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through images in the folder
for frame_number in range(len(bounding_boxes.keys())):  
    # Adjust the range based on the number of frames in your folder

    # Check if the frame number exists in bounding_boxes
    if frame_number not in bounding_boxes:
        print(f"Skipping frame {frame_number} as it is not present in bounding_boxes.")
        continue

    # Read the original image
    image_path = f'{image_folder_path}/traffic/frame_{frame_number:04d}.png'
    image = cv2.imread(image_path)

    # Check if the required keys exist for the current frame
    frame_data = bounding_boxes[frame_number]

    if not isinstance(frame_data, dict) or not all(key in frame_data for key in ('boxes', 'classes', 'names', 'probabilities')):
        print(f"Skipping frame {frame_number} due to missing or invalid data structure.")
        continue

    # Get the original image size
    original_height, original_width, _ = image.shape

    # Draw bounding boxes on the image
    for box, class_id, name, probabilities in zip(
            frame_data['boxes'],
            frame_data['classes'],
            frame_data['names'],
            frame_data['probabilities']
    ):
        x, y, w, h = box

        # Scale the bounding box coordinates based on the original and processed image sizes
        x = int(x * original_width / 500)  # Adjust the denominator based on the resolution used in your VideoWriter
        y = int(y * original_height / 1080)  # Adjust the denominator based on the resolution used in your VideoWriter
        w = int(w * original_width / 1920)  # Adjust the denominator based on the resolution used in your VideoWriter
        h = int(h * original_height / 1080)  # Adjust the denominator based on the resolution used in your VideoWriter

        color = (0, 255, 0)  # Green color for bounding boxes
        thickness = 2

        # Draw bounding box rectangle on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        if class_id < len(probabilities):
            label = f'{name}: {probabilities[class_id]:.2f}'
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Save the annotated frame to the new folder
    output_image_path = f'{output_folder_path}/frame_{frame_number:04d}_annotated.png'
    cv2.imwrite(output_image_path, image)

print("Annotated images saved to the 'detections_with_boxes' folder.")