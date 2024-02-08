import numpy as np

import random
from scipy.stats import beta
import pickle
from utils.sort import Sort
from scipy.stats import beta, ks_2samp
import random
import matplotlib.pyplot as plt
import os
import math

from Bayesian_A_from_pkl import get_prob_from_dist, global_parameterize, fit_beta


def class_counts_from_global(global_data):
    class_counts = [len(value) for value in global_data.values()]
    return class_counts

def get_class_probabilities(class_dictionary):
    class_lengths = [len(data) for data in class_dictionary.values()]
    total_count = sum(class_lengths)
    class_probabilities = [math.sqrt(data) for data in class_lengths]
    while class_probabilities[0] > 0.1:
        class_probabilities[0] = min(0.09, class_probabilities[0])
        datasum = sum(class_probabilities)
        class_probabilities = [data / datasum for data in class_probabilities]
    return class_probabilities

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
        class_pvals = []
        for i in range(m.shape[1]):
            # Perform the KS test for each column
            ks_statistic, p_value = ks_2samp(m[:, i], object_history[:, i])
    
            # Append the p-value to the pvals list
            class_pvals.append(p_value)
        pvalsum = sum(class_pvals)
        class_pvals = [v / pvalsum for v in class_pvals]
        class_pval = max(class_pvals)
        pvals.append(class_pval)
    pvals = [pvals[i] * class_probabilities[i] for i in range(len(pvals))]
    pvalsum = sum(pvals)
    pvals = [round(v / pvalsum,3) for v in pvals]
    
    predicted_class = np.argmax(pvals)
    predicted_class -= 1

    return predicted_class

def predict_detection_class(distribution_parameters, new_features, class_probabilities):
    """Inputs: 
        global class distribution parameters
        new features
        global class probabilities
       Outputs:
        predicted class
       """
        
    # Calculate the pdfs of the new detection using dist parameters
    pvals = []
    for class_index in range(len(distribution_parameters.keys())):
        class_index -= 1
        feature_distribution = distribution_parameters[class_index]
        p = []
        for index, (a, b, loc, scale) in enumerate(feature_distribution):
            probability = beta.pdf(new_features[index], a, b, loc=loc, scale=scale)
            # Add a little so none multiplied by 0
            p.append(probability)
        max_val = 0
        for v in p:
            if v > max_val:
                max_val = v
        pvals.append(max_val)
    pvals = [pvals[i] * class_probabilities[i] for i in range(len(pvals))]
    pvalsum = sum(pvals)
    pvals = [round(v / pvalsum,3) for v in pvals]
    
    predicted_class = np.argmax(pvals)
    predicted_class -= 1

    return predicted_class

def BayesianB(model_folder):
    
    # Define the path for the parsed dictionary of objects found in frame
    detections_path = os.path.join(model_folder, "sort_results.pkl")
    
    global_path = os.path.join(model_folder, "global_data.pkl")
    
    # Load data from a pickle file
    with open(detections_path, 'rb') as pickle_file:
        loaded_frames_detections = pickle.load(pickle_file)
    
    # Load data from a pickle file
    with open(global_path, 'rb') as pickle_file:
        global_data = pickle.load(pickle_file)
    
    # Get class probabilities from population counts
    class_counts = class_counts_from_global(global_data)
    
    # Get distribution parameters from global data
    distribution_parameters = global_parameterize(global_data)
    
    # Initialize dict of data for each tracked object
    tracked_object_data = {}
    
    frame_data = {}
    
    frame_count = len(loaded_frames_detections.keys())
    
    prev_chkpt = 0
    
    # Iterate through every detection in the dictionary
    for frame_num, (frame, detections) in enumerate(loaded_frames_detections.items()):
    
        # Calculate percent with more control over formatting
        percent_unrounded = frame / frame_count
        percent_str = round(percent_unrounded * 100)
    
        # Print progress
        if percent_str != prev_chkpt:
            print(f"Frames: {percent_str}%")
            prev_chkpt = percent_str
        
        # Get class probabilites
        class_probabilities = get_class_probabilities(global_data)
        
        boxes = []
        cls_probs = []
        track_ids = []
        
        for idx in range(len(detections['cls_prob'])):
            box = detections['boxes'][idx,:]
            track_id = detections['track_id'][idx]
            features = detections['cls_prob'][idx]
            
            # If the object is new
            if track_id not in tracked_object_data.keys():
                predicted_class = predict_detection_class(distribution_parameters, features, class_probabilities)
                # Add class prediction to dictionary
                tracked_object_data[track_id] = {"boxes":[box],\
                                                 "class":[predicted_class],\
                                                 'probabilities':features,\
                                                 'frames': [frame_num]}
    
                # Add +1 to population counts for proper class
                class_counts[predicted_class+1] += 1
                
                global_data[predicted_class] = np.vstack([global_data[predicted_class], features])
            
            # If the object has been seen before
            else:
                # Append probability data to the array
                tracked_object_data[track_id]['probabilities'] = np.vstack([tracked_object_data[track_id]['probabilities'], features])
    
    
                # Predict the class of the object based on all data
                predicted_class = predict_tracker_class(global_data, \
                                         tracked_object_data[track_id]['probabilities'], \
                                             class_probabilities)
                
                # Check if predicted class matches the prediction from previous frames in dict
                previous_class = tracked_object_data[track_id]['class'][-1]
                
    
                class_counts[predicted_class] += 1  # Add to new class
    
                # Update tracked object class dictionary
                tracked_object_data[track_id]['boxes'] = np.vstack([tracked_object_data[track_id]['boxes'], box])
                tracked_object_data[track_id]['class'].append(predicted_class)
                tracked_object_data[track_id]['frames'].append(frame_num)
                
                global_data[predicted_class] = np.vstack([global_data[predicted_class], features])
                
            boxes.append(box)
            cls_probs.append(predicted_class)
            track_ids.append(track_id)
            
        frame_data[frame_num] = {'boxes':boxes,\
                                    'class':cls_probs,\
                                    'track_id':track_ids}
            
            
    # Save the dictionary to a .pkl file
    with open(os.path.join(model_folder, "BayesB_tracked_objectdata.pkl"), 'wb') as file:
        pickle.dump(tracked_object_data, file)
        
    # Save the dictionary to a .pkl file
    with open(os.path.join(model_folder, "BayesB_for_metrics.pkl"), 'wb') as file:
        pickle.dump(frame_data, file)

if __name__ == "__main__":
    BayesianB(r"C:\Users\keela\Coding\Models\FinalResults\MLE_Softmax")