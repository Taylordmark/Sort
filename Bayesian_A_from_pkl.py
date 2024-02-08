import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta
import pickle
import os
import concurrent.futures
import math
from fit_beta_cython import fit_beta_cython


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
    class_probabilities = [math.sqrt(data) for data in class_lengths]
    datasum = sum(class_probabilities)
    class_probabilities = [data / datasum for data in class_probabilities]
    while class_probabilities[0] > 0.1:
        class_probabilities[0] = min(0.09, class_probabilities[0])
        datasum = sum(class_probabilities)
        class_probabilities = [data / datasum for data in class_probabilities]
    return class_probabilities

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
        print("Fit error, returning previous parameters")

def local_parameterize_CPU(history, last_params):
    col_parameters = []
    for i, f in enumerate(history.T):
        try:
            a, b, loc, scale = fit_beta(f)
        except:
            a, b, loc, scale = last_params[i]
        col_parameters.append([a, b, loc, scale])
    return col_parameters
             

def local_parameterize_GPU(history):
    col_parameters = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the fit_beta_cython function to each column of the history
        results = executor.map(fit_beta_cython, history.T)

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
    SavePath= os.path.join(folder_path, "BayesA_for_metrics.pkl")
    dict_path = os.path.join(folder_path, "sort_results.pkl")

    # Load detections from a pickle file
    with open(dict_path, 'rb') as pickle_file:
        loaded_frames_detections = pickle.load(pickle_file)
        
    not_bayesed_predictions = []
    for f in loaded_frames_detections.values():
        for d in f['cls_prob']:
            not_bayesed_predictions.append(d)
    
    global_data_path = os.path.join(folder_path, "global_data.pkl")
    
    # Load data from a pickle file
    with open(global_data_path, 'rb') as pickle_file:
        global_data = pickle.load(pickle_file)

    # Calculate global distribution parameters
    print("Parameterizing")
    distribution_parameters = global_parameterize(global_data)
    print("Parameterized")

    prev_chkpt = 0

    all_predictions = {}
    classes_to_reparameterize = set()  # Track classes with new data
    
    frame_count = len(loaded_frames_detections.keys())
    
    # For each frame in the detections dict
    for frame, value in loaded_frames_detections.items():

        class_probabilities = get_class_probabilities(global_data)
        
        # Calculate percent with more control over formatting
        percent_unrounded = frame / frame_count
        percent_str = "{:.0f}".format(percent_unrounded * 100)  # Format to 1 decimal place
    
        # Print progress
        if percent_str != prev_chkpt:
            print(f"Frames: {percent_str}%")
            prev_chkpt = percent_str
            
        
        classes = []
        # For each box
        for detection in range(len(value['boxes'])):
            
            # Get probabilities
            features = value['cls_prob'][detection]

            # Calculate the pdfs of the new detection using dist parameters
            pdfs = []
            for class_index, feature_distribution in distribution_parameters.items():
                p = 1
                for index, (a, b, loc, scale) in enumerate(feature_distribution):
                    probability = beta.pdf(features[index], a, b, loc=loc, scale=scale)
                    # Add a little so none multiplied by 0
                    probability += 0.001
                    p *= probability
                pdfs.append(p)
            
            # Multiply by class probabilities
            pdfsum = sum(pdfs)
            pdfs = [i / sum(pdfs) for i in pdfs]
            pdfs = [round(i, 3) for i in pdfs]
            
            pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]
            
            pdfs[0] = min(pdfs[0], .1)
            
            pdfsum = sum(pdfs)
            pdfs = [i / sum(pdfs) for i in pdfs]
            
            # Return predicted class adjusted for list / class dict index differences
            prediction = np.argmax(pdfs) - 1

            # Add data to where it was predicted to belong
            global_data[prediction] = np.vstack((global_data[prediction], features))
            classes_to_reparameterize.add(prediction)  # Mark class for reparameterization
            
            classes.append(prediction)

        # Reparameterize classes with new data after the frame
        if frame % 50 == 0:
            print(classes_to_reparameterize)
            for class_to_reparam in classes_to_reparameterize:
                distribution_parameters[class_to_reparam] = local_parameterize_CPU(global_data[class_to_reparam], distribution_parameters[class_to_reparam])
            classes_to_reparameterize.clear()  # Reset set for the next frame
        
        # Append boxes and prediction to all_predictions
        all_predictions[frame] = {'boxes': value['boxes'],\
                                  'class': classes,\
                                  'track_id':value['track_id']}

    
    # Save the dictionary to a .pkl file
    with open(SavePath, 'wb') as file:
        pickle.dump(all_predictions, file)

if __name__ == "__main__":
    folder_path = r"C:\Users\keela\Coding\Models\FinalResults\BCE_Sigmoid"
    BayesianA(folder_path)