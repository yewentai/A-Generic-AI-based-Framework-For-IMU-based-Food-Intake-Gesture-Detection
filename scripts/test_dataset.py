import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pickle
from datasets import IMUDataset
import numpy as np
from collections import Counter

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


# Initialize dataset
dataset_left = IMUDataset(X_L, Y_L, sequence_length=128)
dataset_right = IMUDataset(X_R, Y_R, sequence_length=128)

# Compute counts for left hand
all_labels_left = np.concatenate(dataset_left.labels)  # Flatten all labels
left_counts = Counter(all_labels_left)
print("Left Hand Label Distribution:")
print(left_counts)

# Compute counts for right hand
all_labels_right = np.concatenate(dataset_right.labels)
right_counts = Counter(all_labels_right)
print("\nRight Hand Label Distribution:")
print(right_counts)
