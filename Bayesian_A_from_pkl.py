import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta
import pickle
import os

# Define the path for the parsed dictionary of objects found in frame
dict_path = r"C:\Users\keela\Documents\Basic_BCE\initial_detections.pkl"

# Load data from a pickle file
with open(dict_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)

num_classes=10

def get_prob_from_dist(x, parameters):
  """Returns the probability of a number being from a right skewed distribution using all three parameters in a list: shape, scale, and loc, respectively.

  Args:
    x: The number whose probability you want to calculate.
    parameters: A list of the values for a, b, lc, scale respectively

  Returns:
    The probability of the number being from the right skewed distribution.
  """

  # Calculate the probability of the number being from the gamma distribution.
  probability = beta.pdf(x, parameters[0], parameters[1], loc=parameters[2], scale=parameters[3])

  return probability

def get_class_probabilities(class_dictionary):
    total_population = sum(class_dictionary.values())
    class_probabilities = {int(key): value / total_population for key, value in class_dictionary.items()}
    return class_probabilities

def create_plots(no_bayes_data, bayes_data):
    # Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first histogram
    axs[0].hist(no_bayes_data, bins=20, alpha=0.5, label="Data 1")
    
    # Plot the second histogram
    axs[1].hist(bayes_data, bins=20, alpha=0.5, label="Data 2")
    
    # Add labels and title
    axs[0].set_xlabel("Values")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Not Bayesed Distribution")
    axs[1].set_xlabel("Values")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Bayesed Distribution")
    
    # Add legend
    plt.legend(loc="upper right")
    
    # Tight layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Define the labels for each class
class_dictionary = {}

for dict_class in range(num_classes):
    name = str(dict_class)
    amt = int(np.log(random.random() * 5000))
    class_dictionary[name] = amt
    
# Calculate probabilities based on the log-transformed population percentages
class_probabilities = get_class_probabilities(class_dictionary)

# Load detections from a pickle file
with open(dict_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)
    
not_bayesed_predictions = []
for f in loaded_frames_detections.values():
    for d in f['probabilities']:
        not_bayesed_predictions.append(d)
        
total_classes = len(loaded_frames_detections[0]['probabilities'][0])

global_data = {i: np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]) for i in range(10)}



# This part will not exist in the working code but I need to fill
# the global data with something for the rest of it to compare to

# Checks if any dict values are > 30, returns True or False

def all_values_length_gt_0(data_dict):
    """
    Checks if all values in the dictionary are np arrays with no elements between 0 and 30 (exclusive).

    Args:
        data_dict: A dictionary where values are np arrays.

    Returns:
        True if all values have no elements > 30, False otherwise.
    """
    for value in data_dict.values():
        if len(value) > 30:
            return False
        else:
            return True


# Calculates and returns all global distribution parameters
def global_parameterize(data_dict):
    """
    Recalculates and returns all global distribution parameters.

    Args:
        data_dict: A dictionary where values are np arrays.

    Returns:
        A list of lists containing distribution parameters for each column.
    """
    # Initialize an empty list of lists to store parameters
    distribution_parameters = []

    # For each class in the data dictionary
    for k, history in data_dict.items():
        # Initialize an empty list to store parameters for the current class
        col_parameters = []

        # Iterate through each column of the data matrix
        for col in range(len(history[0])):
            # Extract data for the current column
            data = history[:, col]

            # Calculate distribution statistics
            a, b, loc, scale = beta.fit(data)

            # Store parameters for the current column
            col_parameters.append([a, b, loc, scale])

        # Append parameters for the current class to the main list
        distribution_parameters.append(col_parameters)

    return distribution_parameters

# Recalculates and returns a local distribution parameter
def local_parameterize(history):
    """
    Recalculates and returns distribution parameters for a single history.

    Args:
        history: A numpy array representing a single history.

    Returns:
        A list containing distribution parameters for each column.
    """
    # Initialize an empty list to store parameters
    col_parameters = []

    # Iterate through each column of the history
    for col in range(len(history[0])):
        # Extract data for the current column
        data = history[:, col]

        # Calculate distribution statistics
        a, b, loc, scale = beta.fit(data)

        # Store parameters for the current column
        col_parameters.append([a, b, loc, scale])

    return col_parameters

print("Filling globals")

while all_values_length_gt_0(global_data):
    for frame, result in loaded_frames_detections.items():
        for box_num, box in enumerate(result['probabilities']):
            fake_prediction = np.argmax(result['probabilities'][box_num])
            global_data[fake_prediction] = np.vstack((global_data[fake_prediction], box))

print("Globals filled")

# Make empty lists a list of zeros for now
for value in loaded_frames_detections.values():
    if len(value) == 0:
        value = [0 for i in range(total_classes)]

# Calculate global distribution parameters
print("Parameterizing")
distribution_parameters = global_parameterize(global_data)
print("Parameterized")


# Now back to our regularly scheduled program

all_predictions = []
# For each frame in the detections dict
for frame, value in loaded_frames_detections.items():
    # For each detection in the frame
    for detection in range(len(value['boxes'])):
        new_detection = value['probabilities'][detection]
        
        
        # Calculate the pdfs of the new detection using dist parameters
        pdfs = []
        for feature_distribution in distribution_parameters:
            p = 1
            for index, (a, b, loc, scale) in enumerate(feature_distribution):
                probability = beta.pdf(new_detection[index], a, b, loc=loc, scale=scale)
                # Add a little so none multiplied by 0
                probability += 0.001
                p *= probability
            pdfs.append(p)

        # Multiply by population probabilities
        pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]

        prediction = pdfs.index(max(pdfs))
        
        # Add data to where it was predicted to belong
        global_data[prediction] = np.vstack((global_data[prediction], new_detection))
        
        
        # Recalculate the respective parameters
        distribution_parameters[prediction] = local_parameterize(global_data[prediction])
        all_predictions.append(prediction)
    print(f"Processing frames {round(frame/len(loaded_frames_detections.keys()), 3)}")


detections_path = r"C:\Users\keela\Documents\Basic_BCE\BayesA.pkl"
# Save the dictionary to a .pkl file
with open(detections_path, 'wb') as file:
    pickle.dump(all_predictions, file)
    
    
    
