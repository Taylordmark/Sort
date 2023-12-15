import numpy as np

import random
from scipy.stats import beta
import pickle
from sort import Sort
from scipy.stats import beta, ks_2samp
import random
import matplotlib.pyplot as plt
import os

from A_probability_from_pdfs import get_prob_from_dist, global_parameterize




def probabilities_from_populations(population_counts):
    total_population = sum(population_counts.values())
    class_probabilities = {key: value / total_population for key, value in population_counts.items()}
    return class_probabilities

def parse_data_from_detections(detections):
    xywh_boxes, probabilities, classes, names = detections.values()
    xyxy_boxes = []
    for b in xywh_boxes:
        b = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
        xyxy_boxes.append(b)
    boxes = np.array(xyxy_boxes)
    probabilities = np.array(probabilities)
    classes = np.array(classes)
    
    return boxes, probabilities, classes, names

def find_matching_box_index(boxes, bbox):
    for i, box in enumerate(boxes):
        if np.allclose(box, bbox, rtol=.1):
            return i
    return None
        
def predict_tracker_class(global_data, object_history, class_probabilities):
    
    # KS test all classes, features and get list of p values
    pvals = []
    for c in global_data:
        pval = 1 
        for i in range(c.shape[1]):
            # Perform the KS test for each column
            ks_statistic, p_value = ks_2samp(c[:, i], object_history[:, i])
    
            # Append the p-value to the pvals list
            pval *= p_value
        pvals.append(pval)
    pvals = [pvals[i] * class_probabilities[i] for i in range(len(pvals))]
    
    predicted_class = pvals.index(max(pvals))

    return predicted_class

def predict_detection_class(distribution_parameters, new_distribution, class_probabilities):
    """Inputs: 
        global class distribution parameters
        new distribution parameters
        global class probabilities
       Outputs:
        predicted class
       """
        
    pdfs = []
    for row in distribution_parameters:
        p = 1
        for index, (a, b, loc, scale) in enumerate(row):
            probability = beta.pdf(new_distribution[index], a, b, loc=loc, scale=scale)
            p *= probability
        pdfs.append(p)

    # Multiply by population probabilities
    pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]

    prediction = pdfs.index(max(pdfs))
    return prediction

def create_global_data(tracked_object_data, tracked_object_classes, num_classes):
  """
  Creates and initializes the global data structure based on tracked object data and classes.

  Args:
    tracked_object_data: A dictionary containing tracked object data for each object ID.
    tracked_object_classes: A dictionary containing the predicted class for each tracked object.
    num_classes: The total number of object classes.

  Returns:
    global_data: An np.array containing the global data structure, initialized with tracked object data.
  """

  global_data = np.full((num_classes, 1), 1)  # Initialize with empty data for all classes

  # Iterate through tracked object data
  for track_id, data in tracked_object_data.items():
    predicted_class = tracked_object_classes.get(track_id, None)  # Get predicted class
    if predicted_class:
      # Update global data for the predicted class
      global_data[predicted_class] = np.vstack((global_data[predicted_class], data))

  return global_data



# Define the path for the parsed dictionary of objects found in frame
dict_path = '/home/taylordmark/Thesis/Sort/parsed_data_dict.pkl'

# Define path for list of classes model trained on
classes_path = 'C:/Users/keela/Documents/Sort/yolo-cls-traffic_only.txt'

# Load data from a pickle file
with open(dict_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)


# Get all classes
with open(classes_path, 'r') as classes_file:
    classes = classes_file.readlines()
classes = [c.replace("\n", "") for c in classes]
num_classes = len(classes)

# Generate fake distributions for previous class instances (fix later)
population_counts = {}

for c in range(num_classes):
    population_counts[c] = int(np.log(float((random.randint(1,10000))**5)))

# Get class probabilities from population counts
class_probabilities = probabilities_from_populations(population_counts)

# Initialize SORT tracker
mot_tracker = Sort()

# Initialize dict of data for each tracked object
tracked_object_data = {}

# Initialize dict of data for predicted object class
tracked_object_classes = {}

# Iterate through every detection in the dictionary
for frame, detections in loaded_frames_detections.items():
    boxes, probabilities, classes, names = parse_data_from_detections(detections)
    
    
    # Update SORT with boxes
    trackers = mot_tracker.update(boxes)
    
    # If there is a tracked object
    if len(trackers) > 0:
        # Iterate through tracked objects
        for o in trackers:
            bbox = o[:4]
            track_id = o[-1]
            
            # If the object is new
            if track_id not in tracked_object_data.keys():
                # Find which of the current bboxes match
                index = find_matching_box_index(boxes, bbox)

                # Get data from the proper index
                features = np.array(probabilities[index])

                # Set track data to features list
                tracked_object_data[track_id] = features
                
                # Fill global_data
                global_data = create_global_data(tracked_object_data, tracked_object_classes, num_classes)

                # Get predicted class from pdfs
                predicted_class = predict_detection_class(global_data, features, class_probabilities)

                # Add class prediction to dictionary
                tracked_object_classes[predicted_class] = predicted_class

                # Add +1 to population counts for proper class
                population_counts[predicted_class] += 1
            
            # If the object has been seen before
            else:
                # Append probability data to the list
                tracked_object_data[track_id] = np.vstack(np.array(probabilities[index]))
                
                # Fill global_data
                global_data = create_global_data(tracked_object_data, tracked_object_classes, num_classes)

                # Predict the class of the object based on all data
                predicted_class = \
                    predict_detection_class(global_data, \
                                             tracked_object_data[track_id], \
                                                 class_probabilities)
                
                # Check if predicted class matches the prediction from previous frames in dict
                previous_class = tracked_object_classes.get(track_id, None)  # Get previously predicted class, if any
                
                if predicted_class != previous_class:
                    # Update population counts
                    if previous_class:
                        population_counts[previous_class] -= 1  # Subtract from previous class
                    population_counts[predicted_class] += 1  # Add to new class

                    # Update tracked object class dictionary
                    tracked_object_classes[track_id] = predicted_class

            
        #print(f"{frame}:{trackers}\n")
    else:
        print(f"Nothing tracked in {frame}")
