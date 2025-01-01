import pickle
from utils import IMUDataset
import numpy as np

# Load data
X_path = "./dataset/FD/FD-I/X_L.pkl"
Y_path = "./dataset/FD/FD-I/Y_L.pkl"
with open(X_path, "rb") as f:
    X = pickle.load(f)
with open(Y_path, "rb") as f:
    Y = pickle.load(f)

# Create the full dataset
full_dataset = IMUDataset(X, Y)
unique_subjects = np.unique(full_dataset.subject_indices)

# Print the number of folds
print("Number of folds:", len(unique_subjects))

# Print the number of samples
print("Number of samples:", len(full_dataset))
