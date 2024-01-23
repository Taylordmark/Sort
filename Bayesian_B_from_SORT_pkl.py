import numpy as np

import random
from scipy.stats import beta
import pickle
from sort import Sort
from scipy.stats import beta, ks_2samp
import random
import matplotlib.pyplot as plt
import os

from Bayesian_A_from_pkl import get_prob_from_dist, global_parameterize, fit_beta


def class_counts_from_global(global_data):
    class_counts = [len(value) for value in global_data.values()]
    return class_counts

def parse_data_from_detections(detections):
    xywh_boxes, probabilities = detections.values()
    xyxy_boxes = []
    for b in xywh_boxes:
        b = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
        xyxy_boxes.append(b)
    boxes = np.array(xyxy_boxes)
    probabilities = np.array(probabilities)
    
    return boxes, probabilities

def find_matching_box_index(boxes, bbox):
    # Calculate distances between all SORT boxes and all detection boxes
    boxes_centers = np.stack([boxes[:, 1] - boxes[:, 0], boxes[:, 3] - boxes[:, 2]], axis=1)
    bbox_center = np.array([bbox[1] - bbox[0], bbox[3] - bbox[2]])
    distances = boxes_centers - bbox_center

    # Calculate Euclidean distances
    distances = np.linalg.norm(distances, axis=1)

    # Find the indices of the detection boxes with the minimum distances
    min_distance_indices = np.argmin(distances, axis=0)

    return min_distance_indices

def predict_tracker_class(global_data, object_history, class_probabilities):
    
    # KS test all classes, features and get list of p values
    pvals = []
    for c, m in global_data.items():
        pval = 1 
        for i in range(m.shape[1]):
            # Perform the KS test for each column
            ks_statistic, p_value = ks_2samp(m[:, i], object_history[:, i])
    
            # Append the p-value to the pvals list
            pval *= p_value
        pvals.append(pval)
    pvals = [pvals[i] * class_probabilities[i] for i in range(len(pvals))]
    
    predicted_class = pvals.index(max(pvals))

    return predicted_class

def predict_detection_class(distribution_parameters, new_features, class_probabilities):
    """Inputs: 
        global class distribution parameters
        new features
        global class probabilities
       Outputs:
        predicted class
       """
        
    pdfs = []
    for class_id in distribution_parameters.keys():
        p = 1
        for feature_parameters, new_feature in zip(distribution_parameters[class_id], new_features):
            a, b, loc, scale = feature_parameters
            probability = beta.pdf(new_feature, a, b, loc=loc, scale=scale)
            p *= probability

        pdfs.append(p)
    
    # Multiply by population probabilities
    pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]
    
    prediction = np.argmax(pdfs) - 1
    return prediction

# Define the model folder path
model_folder = r"C:\Users\keela\Documents\Models\LastMinuteRuns\Small_MLE"
# Define the path for the parsed dictionary of objects found in frame
detections_path = r"C:\Users\keela\Documents\Models\LastMinuteRuns\Small_MLE\000f8d37-d4c09a0f_initial_detections.pkl"


global_path = os.path.join(model_folder, "global_data.pkl")

# Define path for list of classes model trained on
classes_path = r"C:\Users\keela\Documents\Prebayesian\class_list_traffic.txt"

# Load data from a pickle file
with open(detections_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)

# Load data from a pickle file
with open(global_path, 'rb') as pickle_file:
    global_data = pickle.load(pickle_file)

# Get all classes
with open(classes_path, 'r') as classes_file:
    classes = classes_file.readlines()
classes = [c.replace("\n", "") for c in classes]
num_classes = len(classes)

# Get class probabilities from population counts
class_counts = class_counts_from_global(global_data)

# Get distribution parameters from global data
distribution_parameters = global_parameterize(global_data)

# Initialize SORT tracker
mot_tracker = Sort()

# Initialize dict of data for each tracked object
tracked_object_data = {}

track_ids = []

# Iterate through every detection in the dictionary
for frame_num, (frame, detections) in enumerate(loaded_frames_detections.items()):
    boxes, probabilities = parse_data_from_detections(detections)

    # Update SORT with boxes
    trackers = mot_tracker.update(boxes)
    
    # Get predicted class from pdfs
    class_probabilities = [value / sum(class_counts) for value in class_counts]
    # Combat 'unknown' object weight inflation
    class_probabilities[0] = min(class_probabilities[0], 0.25)
    sum_of_probs = sum(class_probabilities)
    # Renormalize
    class_probabilities = [value / sum_of_probs for value in class_counts]
    
    # If objects are tracked
    if len(trackers) > 0:
        # Iterate through tracked objects
        for o in trackers:
            bbox = o[:4]
            track_id = o[-1]
            
            index = find_matching_box_index(boxes, bbox)
            
            new_probabilities = np.array(probabilities[index][:])
            
            # If the object is new
            if track_id not in tracked_object_data.keys():                
                # Get data from the proper index
                features = probabilities[index]

                predicted_class = predict_detection_class(distribution_parameters, features, class_probabilities)

                # Add class prediction to dictionary
                tracked_object_data[track_id] = {"boxes":[bbox],\
                                                 "class":[predicted_class],\
                                                 'probabilities':new_probabilities,\
                                                 'frames': [frame_num]}

                # Add +1 to population counts for proper class
                class_counts[predicted_class+1] += 1
            
            # If the object has been seen before
            else:
                # Append probability data to the array
                tracked_object_data[track_id]['probabilities'] = np.vstack([tracked_object_data[track_id]['probabilities'], new_probabilities])


                # Predict the class of the object based on all data
                predicted_class = \
                predict_tracker_class(global_data, \
                                         tracked_object_data[track_id]['probabilities'], \
                                             class_probabilities)
                
                # Check if predicted class matches the prediction from previous frames in dict
                previous_class = tracked_object_data[track_id]['class'][-1]
                
                if predicted_class != previous_class:
                    # Update population counts
                    if previous_class:
                        class_counts[previous_class] -= 1  # Subtract from previous class
                    class_counts[predicted_class] += 1  # Add to new class

                # Update tracked object class dictionary
                tracked_object_data[track_id]['boxes'] = np.vstack([tracked_object_data[track_id]['boxes'], bbox])
                tracked_object_data[track_id]['class'].append(predicted_class)
                tracked_object_data[track_id]['frames'].append(frame_num)
            
    else:
        print(f"Nothing tracked in {frame}")

# Save the dictionary to a .pkl file
with open(os.path.join(model_folder, "BayesB.pkl"), 'wb') as file:
    pickle.dump(tracked_object_data, file)
