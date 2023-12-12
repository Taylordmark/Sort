import numpy as np
import random
from scipy.stats import beta
from scipy.stats import ks_2samp

import numpy as np
import pickle
from sort import Sort
from scipy.stats import beta, ks_2samp

# Other functions and definitions...

def predict_class_for_object(global_data, class_probabilities, sorted_data):
    predicted_classes = []
    object_history = np.full((1, num_classes), 1)

    # Initialize SORT tracker
    mot_tracker = Sort()

    for frame, value in sorted_data.items():
        # Get the bounding boxes and scores from the current frame
        detections = np.array(value['boxes'])
        scores = np.array(value['probabilities'])

        # Use SORT to track the objects
        trackers = mot_tracker.update(detections)

        # Iterate through each detected object
        for d, detection in enumerate(trackers):
            # Extract the bounding box and score
            bbox = detection[:4]
            score = scores[d]

            # Use the bbox and score to get the predicted class
            # ... (your logic for class prediction based on bbox and score)

            # Add data to where it was predicted to belong
            global_data[predicted_class] = np.vstack((global_data[predicted_class], score))

            # Recalculate the respective parameters
            global_data[predicted_class] = create_distribution_parameters_list(num_classes, global_data[predicted_class])

    return predicted_classes

# Rest of the code...


def get_class_probabilities(population_counts):
    total_population = sum(population_counts.values())
    class_probabilities = {key: value / total_population for key, value in population_counts.items()}
    return class_probabilities

def get_parameters_from_list(list_of_numbers):
    a, b, loc, scale = beta.fit(list_of_numbers)
    parameters = [a, b, loc, scale]
    return parameters

def create_distribution_parameters_list(num_classes, object_type_distributions):
    # Create an empty list of distribution parameters
    distribution_parameters = []

    # For each class, feature set dist parameters to fitted beta dist of associated data
    for row in range(num_classes):
        row_parameters = []
        for col in range(num_classes):
            data = object_type_distributions[row][:, col]

            # Get parameters
            parameters = get_parameters_from_list(data)

            # Store parameters in a dictionary
            row_parameters.append(parameters)
        distribution_parameters.append(row_parameters)    
    return distribution_parameters

def predict_class_for_detection(distribution_parameters, new_distribution, class_probabilities):
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
            # Add a little so none multiplied by 0, shouldn't be needed with real data
            probability += 0.001
            p *= probability
        pdfs.append(p)

    # Multiply by population probabilities
    pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]

    prediction = pdfs.index(max(pdfs))
    return prediction

def predict_class_for_object(global_data, class_probabilities):
    # Randomly select a "true" class based on the probabilities
    true_class = random.choices(list(class_probabilities.keys()), list(class_probabilities.values()))[0]
    predicted_classes = []
    object_history = np.full((1, num_classes), 1)

    # For every frame generate a prediction for the object and add to data
    for frame in range(100):
        # Generate numbers for all list indices
        object_features = [random.random() for _ in range(len(list(class_probabilities.keys())))]
        # Add a little to the true class index
        # .25 makes the probability of true class being true class about 50/50
        object_features[true_class] += .25
        # Make one of the other classes be mildly indicitive of the true class
        object_features[true_class - 2] -= 0.1
        # Normalize the values
        features_sum = sum(object_features)
        object_features = [x / features_sum for x in object_features]
        # Stack object features with previous values for predicted class
        object_features = np.array(object_features)
        object_history = np.vstack((object_history, object_features))
    
        # KS test all classes, features and get list of p values
        pvals = []
        for clas in global_data:
            pval = 1 
            for i in range(clas.shape[1]):
                # Perform the KS test for each column
                ks_statistic, p_value = ks_2samp(clas[:, i], object_history[:, i])
        
                # Append the p-value to the pvals list
                pval *= p_value
            pvals.append(pval)
        pvals = [pvals[i] * class_probabilities[i] for i in range(len(pvals))]
        
        predicted_classes.append(pvals.index(max(pvals)))

    return predicted_classes

# Define the "new detection" probability distribution

# Person-esque prediction
new_detection = np.array([0.6, 0.2, 0.2, 0.0, 0.0])

# Traffic light-esque prediction
# new_detection = np.array([.0, .1, .1, .3, .5])

# Bicycle-esque prediction
# new_detection = np.array([.1, .6, .0, .2, .1])

# Define class  labels
class_dictionary = {
    0: "person",  
    1: "bicycle",  
    2: "car",     
    3: "motorcycle",  
    4: "traffic light"}\

# Define the population percentages for each class as log of estimated counts
population_counts = {
    0: np.log(501997),  
    2: np.log(24300),     
    3: np.log(11261),  
    4: np.log(1778)}

# Calculate probabilities based on the log-transformed population percentages
class_probabilities = get_class_probabilities(population_counts)
print(class_probabilities.keys())

# Get number of classes
num_classes = len(population_counts)

# Create np arrays to track distributions for all classes
object_type_distributions = [np.full((1, num_classes), 1) \
                             for _ in range(len(class_dictionary))]

# Generate data for x number of detections
detection_count = 1000
object_type_distributions = generate_data(num_classes, object_type_distributions, detection_count)
print(object_type_distributions[0])

# Create distribution parameters list and predict for new detection
# distribution_parameters = create_distribution_parameters_list(num_classes, object_type_distributions)
# detection_prediction = predict_class_for_detection(distribution_parameters, new_detection, class_probabilities)

predictions = predict_class_for_object(object_type_distributions, class_probabilities)
print(predictions)