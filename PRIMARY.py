from model_user import use_model
from Bayesian_A_from_pkl import BayesianA
from SORT_from_pkl import SORT_initial_detections

folder_path=r"C:\Users\keela\Documents\Models\binary_crossentropy"
image_folder=r"C:\Users\keela\Documents\Video Outputs\0000f77c-6257be58\frames"

use_model(folder_path, image_folder)
BayesianA(folder_path)
SORT_initial_detections(folder_path)
