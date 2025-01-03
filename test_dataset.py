import pickle
from utils import IMUDataset
import numpy as np

import pickle
import numpy as np
from utils import IMUDataset

# Load data
X_L_path = "./dataset/FD/FD-I/X_L.pkl"
Y_L_path = "./dataset/FD/FD-I/Y_L.pkl"
X_R_path = "./dataset/FD/FD-I/X_R.pkl"
Y_R_path = "./dataset/FD/FD-I/Y_R.pkl"

with open(X_L_path, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)  # Convert to NumPy array
with open(Y_L_path, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_path, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_path, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Debug: Inspect the shapes and types
print("Type of X_L:", type(X_L))
print("Shape of X_L:", X_L.shape)
print("Type of X_R:", type(X_R))
print("Shape of X_R:", X_R.shape)

# Concatenate the left and right data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset
full_dataset = IMUDataset(X, Y)
unique_subjects = np.unique(full_dataset.subject_indices)

# Print the number of folds
print("Number of folds:", len(unique_subjects))

# Print the number of samples
print("Number of samples:", len(full_dataset))
