import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta
import pickle
import os
import concurrent.futures


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
    class_lengths = [len(data) for data in class_dictionary.values()]
    total_count = sum(class_lengths)
    class_probabilities = [data / total_count for data in class_lengths]
    return class_probabilities

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

def global_parameterize(data_dict):
    """
    Recalculates and returns all global distribution parameters.

    Args:
        data_dict: A dictionary where values are np arrays.

    Returns:
        A list of lists containing distribution parameters for each column.
    """
    # Initialize an empty list of lists to store parameters
    distribution_parameters = {}

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
        distribution_parameters[k] = col_parameters

    return distribution_parameters

def fit_beta(data):
    try:
        return beta.fit(data)
    except:
        print("FitError: Returning default parameters.")
        # Return default parameters or handle as needed
        return [1, 100, 0, 0]


def local_parameterize(history):
    col_parameters = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the fit_beta function to each column of the history
        results = executor.map(fit_beta, history.T)

    # Store parameters for each column
    col_parameters = list(results)

    return col_parameters

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


def BayesianA(folder_path):
    SavePath= os.path.join(folder_path, "BayesA.pkl")
    dict_path = os.path.join(folder_path, "initial_detections.pkl")

    # Load data from a pickle file
    with open(dict_path, 'rb') as pickle_file:
        loaded_frames_detections = pickle.load(pickle_file)

    num_classes=10

    # Load detections from a pickle file
    with open(dict_path, 'rb') as pickle_file:
        loaded_frames_detections = pickle.load(pickle_file)
        
    not_bayesed_predictions = []
    for f in loaded_frames_detections.values():
        for d in f['probabilities']:
            not_bayesed_predictions.append(d)
            
    total_classes = len(loaded_frames_detections[0]['probabilities'][0])

    global_data_path = os.path.join(folder_path, "global_data.pkl")
    
    # Load data from a pickle file
    with open(global_data_path, 'rb') as pickle_file:
        global_data = pickle.load(pickle_file)

    # Calculate global distribution parameters
    print("Parameterizing")
    distribution_parameters = global_parameterize(global_data)
    print("Parameterized")
    
    class_probabilities = get_class_probabilities(global_data)

    frame_count = len(loaded_frames_detections.keys())
    counter = 0
    prev_chkpt = 0

    all_predictions = []
    classes_to_reparameterize = set()  # Track classes with new data

    # For each frame in the detections dict
    for frame, value in loaded_frames_detections.items():
        counter += 1
        percent = round(counter / frame_count, 2)
        if percent > prev_chkpt:
            print(f"\nFrames: {percent}\n")
            prev_chkpt = percent

        # For each detection in the frame
        detection_counter = 0
        detection_total = len(value['boxes'])

        # Limit number of detections processed
        for detection in range(len(value['boxes'])):
            print(f"{detection_counter / detection_total:.2f}")
            detection_counter += 1

            new_detection = value['probabilities'][detection]

            # Calculate the pdfs of the new detection using dist parameters
            pdfs = []
            for class_index, feature_distribution in distribution_parameters.items():
                p = 1
                for index, (a, b, loc, scale) in enumerate(feature_distribution):
                    probability = beta.pdf(new_detection[index], a, b, loc=loc, scale=scale)
                    # Add a little so none multiplied by 0
                    probability += 0.001
                    p *= probability
                pdfs.append(p)

            # Multiply by population probabilities
            
            pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]

            prediction = np.argmax(pdfs) - 1

            # Add data to where it was predicted to belong
            global_data[prediction] = np.vstack((global_data[prediction], new_detection))
            classes_to_reparameterize.add(prediction)  # Mark class for reparameterization

        # Reparameterize classes with new data after the frame
        print(classes_to_reparameterize)
        for class_to_reparam in classes_to_reparameterize:
            distribution_parameters[class_to_reparam] = local_parameterize(global_data[class_to_reparam])
        classes_to_reparameterize.clear()  # Reset for the next frame

        all_predictions.append(prediction)
        print(f"Processing frames {round(frame/len(loaded_frames_detections.keys()), 3)}")

    
    # Save the dictionary to a .pkl file
    with open(SavePath, 'wb') as file:
        pickle.dump(all_predictions, file)

if __name__ == "__main__":
    folder_path = r"C:\Users\keela\Documents\Models\Basic_CCE"
    BayesianA(folder_path)