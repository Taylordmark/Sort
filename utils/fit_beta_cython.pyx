# fit_beta_cython.pyx
import numpy as np
from scipy.stats import beta

def fit_beta_cython(data):
    try:
        return beta.fit(data)
    except:
        print("FitError: Returning default parameters.")
        # Return default parameters or handle as needed
        return [1, 100, 0, 0]
