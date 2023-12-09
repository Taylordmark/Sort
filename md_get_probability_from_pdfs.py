import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta


def generate_data(class_count, object_type_distributions, frames=100):
    for frame in range(frames):
        # Randomly select a "true" class based on the probabilities
        true_class = random.choices(list(class_probabilities.keys()), list(class_probabilities.values()))[0]
        # Generate numbers for all list indices
        object_features = [random.random() for _ in range(class_count)]
        # Add a little to the true class index
        # .25 makes the probability of true class being true class about 50/50
        object_features[true_class] += .25
        # Make one of the other classes be mildly indicitive of the true class
        object_features[true_class - 2] -= 0.1
        # Normalize the values
        features_sum = sum(object_features)
        object_features = [x / features_sum for x in object_features]
        # Get predicted class
        predicted_class = object_features.index((max(object_features)))
        # Stack object features with previous values for predicted class
        object_features = np.array(object_features)
        object_type_distributions[predicted_class] = np.vstack((object_type_distributions[predicted_class], object_features))
    return object_type_distributions

def generate_plot(num_classes, object_type_distributions):
    # Define the number of rows (i) and columns (j) for the panel
    i, j = num_classes, num_classes
    # Create subplots for the i x j panel
    fig, axes = plt.subplots(i, j, figsize=(15, 10))
    # Loop through each combination of columns and create histograms
    for row in range(i):
        for col in range(j):
            # Plot the histogram
            axes[row, col].hist(object_type_distributions[row][:, col], bins=10)  # Adjust the number of bins as needed
            axes[row, col].set_title(f'Column {col + 1} Histogram for Object Type {row + 1}')
            axes[row, col].set_xlabel('Values')
            axes[row, col].set_ylabel('Frequency')

            # Set the x-axis limits to match the maximum value
            axes[row, col].set_xlim(0, max_value)


    # Display the histograms
    plt.tight_layout()
    plt.show()

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

def get_class_probabilities(population_counts):
    total_population = sum(population_counts.values())
    class_probabilities = {key: value / total_population for key, value in population_counts.items()}
    return class_probabilities

# Define the labels for each class
class_dictionary = {
    0: "person",  
    1: "bicycle",  
    2: "car",     
    3: "motorcycle",  
    4: "traffic light"}

# Define the population percentages for each class
population_counts = {
    0: 501997,  
    1: 24892,  
    2: 24300,     
    3: 11261,  
    4: 1778}

# Define the current probability distribution

# Person-esque prediction
new_distribution = np.array([0.6, 0.2, 0.2, 0.0, 0.0])

# Traffic light-esque prediction
# new_distribution = np.array([.0, .1, .1, .3, .5])

# Bicycle-esque prediction
# new_distribution = np.array([.1, .6, .0, .2, .1])




# Log transform initial population values
population_counts = {key: np.log(value) for key, value in population_counts.items()}

# Calculate probabilities based on the log-transformed population percentages
class_probabilities = get_class_probabilities(population_counts)


# Create np arrays to track prediction distributions for class x
object_type_distributions = [np.full((1, len(class_dictionary)), 1) \
                             for _ in range(len(class_dictionary))]


num_classes = len(population_counts)

object_type_distributions = generate_data(num_classes, object_type_distributions, 1000)


distribution_parameters = []
for row in range(num_classes):
    row_parameters = []
    for col in range(num_classes):
        data = object_type_distributions[row][:, col]

        # Calculate distribution statistics
        a, b, loc, scale = beta.fit(data)
        # Store parameters in a dictionary
        row_parameters.append([a, b, loc, scale])
    distribution_parameters.append(row_parameters)


pdfs = []
for row in distribution_parameters:
    p = 1
    for index, (a, b, loc, scale) in enumerate(row):
        probability = beta.pdf(new_distribution[index], a, b, loc=loc, scale=scale)
        # Add a little so none multiplied by 0
        probability += 0.001
        p *= probability
    pdfs.append(p)

# Multiply by population probabilities
pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]

prediction = pdfs.index(max(pdfs))
print(f"\nClass prediction: {class_dictionary[prediction]}")