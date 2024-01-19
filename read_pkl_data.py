import pickle
import cv2
from pathlib import Path
import os
import numpy as np

def generate_video_from_pkl(pkl_file_path, output_video_path, img_path):
    with open(pkl_file_path, 'rb') as pickle_file:
        results_dict = pickle.load(pickle_file)

    # Path to the image folder
    image_folder = Path(img_path)

    # Get the dimensions of the first image in the folder
    first_image = next(image_folder.iterdir())
    first_image = cv2.imread(str(first_image))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    counter = 0
    total_frames = len(os.listdir(image_folder))
    progress = 0

    # Sort image file paths based on filenames
    image_paths = sorted(map(str, image_folder.iterdir()))  # Convert WindowsPath objects to strings

    for frame_num, frame_data in results_dict.items():
        p = float(f"{counter / total_frames:.2f}")
        if p > progress:
            print(p)
            progress = p
        counter += 1
        image_path = image_paths[frame_num]  # Get the corresponding image path

        image = cv2.imread(image_path)  # Use the string path directly
        boxes = frame_data['boxes']
        probabilities = frame_data['probabilities']

        for (xmin, ymin, xmax, ymax), confidence in zip(boxes, probabilities):
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            object_id_and_class = f"{np.argmax(confidence)}: {np.max(confidence):.3f}"
            color = (0, 255, 0)  # Example color (green), you can customize as needed
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, object_id_and_class, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame with bounding boxes to the output video
        out.write(image)

    # Release the VideoWriter object and close the video file
    out.release()

    print(f'Video saved as {output_video_path}')

# Replace 'your_file_path.pkl' and 'your_output_video_path.avi' with the actual paths
generate_video_from_pkl(r'C:\Users\keela\Documents\Models\bce\initial_detections.pkl',\
                         r'C:\Users\keela\Documents\Models\bce\initial_detections.avi',\
                            r"C:\Users\keela\Documents\Video Outputs\0000f77c-6257be58\frames")