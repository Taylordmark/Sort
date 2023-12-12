import json
import ast

file_path = '/remote_home/Thesis/Sort/frames_output copy.txt'


def read_valuestring(valstring):
    valstring = valstring[1:-1]
    valstring = valstring.replace("\n", " ")
    valstring = valstring.replace("array", "")
    valstring = valstring.replace("dtype=float32", "")
    valstring = ast.literal_eval(valstring)

    valstring = {
        'boxes': valstring[0][0],
        'probabilities': valstring[1][0][0],
        'classes': valstring[2],
        'names': valstring[3]
    }
    
    return valstring

# Read data from the file
with open(file_path, 'r') as file:
    data_lines = file.readlines()

# Create a dictionary to store the parsed data
data_dict = {}

# Initialize variables to store key and value
current_key = None
current_value_lines = []

# Iterate through each line
for line in data_lines:
    # Check if the line contains ":"
    if ":" in line:
        # If a key is already set, save the previous key-value pair
        if current_key is not None:
            # Combine the lines into a single string
            value_str = ''.join(current_value_lines).strip()

            data_dict[int(current_key)] = read_valuestring(value_str)

        # Split the line into key and value using the first occurrence of ":"
        current_key, current_value_lines = line.split(':', 1)
        # Reset value for the new key
        current_value_lines = [current_value_lines]
    else:
        # If no ":" is found, add the line to the current value
        current_value_lines.append(line)

    
# Add the last key-value pair to the dictionary
if current_key is not None:
    # Combine the lines into a single string
    value_str = ''.join(current_value_lines).strip()

    # Assign the key-value pair to the dictionary
    data_dict[int(current_key)] = read_valuestring(value_str)


# Define the file path
file_path = '/remote_home/Thesis/Sort/parsed_data_no_planes.txt'

# Save the dictionary to a text file using json.dump
with open(file_path, 'w') as file:
    json.dump(data_dict, file)

print(f"Dictionary saved to {file_path}")