import pandas as pd
import numpy as np
import glob
import os
from collections import Counter
import pickle as pkl
from itertools import groupby
from operator import itemgetter

"""
Process the raw data and annotations for DX-I dataset
"""
# Define paths
path1 = r"dataset/DX-I/Raw data and Annotation/Left hand"
path2 = r"dataset/DX-I/Raw data and Annotation/Right hand"

# Load files for both hands
left_files = glob.glob(os.path.join(path1, "*.csv"))
right_files = glob.glob(os.path.join(path2, "*.csv"))

# Extract subject ID and file path
left_file_info = [(int(os.path.basename(f).split("-")[0]), f, "L") for f in left_files]
right_file_info = [
    (int(os.path.basename(f).split("-")[0]), f, "R") for f in right_files
]

# Combine and sort files by subject ID
file_info = left_file_info + right_file_info
file_info.sort(key=itemgetter(0))

# Group files by subject ID
grouped_files = {k: list(g) for k, g in groupby(file_info, key=itemgetter(0))}

# Initialize storage
data_intotal, target_intotal = [], []
data_left, target_left = [], []
data_right, target_right = [], []

# Process each subject's files
for subject_id, files in grouped_files.items():
    subject_data, subject_target = [], []
    subject_data_L, subject_target_L = [], []
    subject_data_R, subject_target_R = [], []

    for _, f, hand in files:
        df = pd.read_csv(f)
        df = df.drop(["t"], axis=1)
        target = df.pop("anno")

        # Save general subject data
        subject_data.append(df.values)
        subject_target.append(target.values)

        # Separate left and right hand data
        if hand == "L":
            subject_data_L.append(df.values)
            subject_target_L.append(target.values)
        else:
            subject_data_R.append(df.values)
            subject_target_R.append(target.values)

    # Concatenate per subject
    subject_data = np.concatenate(subject_data, axis=0)
    subject_target = np.concatenate(subject_target, axis=0)

    data_intotal.append(subject_data)
    target_intotal.append(subject_target)

    # Process left hand data
    if subject_data_L:
        subject_data_L = np.concatenate(subject_data_L, axis=0)
        subject_target_L = np.concatenate(subject_target_L, axis=0)
        data_left.append(subject_data_L)
        target_left.append(subject_target_L)

    # Process right hand data
    if subject_data_R:
        subject_data_R = np.concatenate(subject_data_R, axis=0)
        subject_target_R = np.concatenate(subject_target_R, axis=0)
        data_right.append(subject_data_R)
        target_right.append(subject_target_R)

# Verify label distribution
for i in range(len(target_intotal)):
    print(f"Subject {i+1}: {Counter(target_intotal[i])}")

# Save combined dataset
os.makedirs("dataset/DX/DX-I", exist_ok=True)
with open("dataset/DX/DX-I/DX_I_X.pkl", "wb") as f:
    pkl.dump(data_intotal, f)
with open("dataset/DX/DX-I/DX_I_Y.pkl", "wb") as f:
    pkl.dump(target_intotal, f)

# Save left hand dataset
with open("dataset/DX/DX-I/X_L.pkl", "wb") as f:
    pkl.dump(data_left, f)
with open("dataset/DX/DX-I/Y_L.pkl", "wb") as f:
    pkl.dump(target_left, f)

# Save right hand dataset
with open("dataset/DX/DX-I/X_R.pkl", "wb") as f:
    pkl.dump(data_right, f)
with open("dataset/DX/DX-I/Y_R.pkl", "wb") as f:
    pkl.dump(target_right, f)

print("Data processing complete. Files saved.")


""""
Process the raw data and annotations for DX-II dataset
"""
# Define paths
path1 = r"dataset/DX-II/Raw data and Annotation/Left hand"
path2 = r"dataset/DX-II/Raw data and Annotation/Right hand"

# Load files for both hands
left_files = glob.glob(os.path.join(path1, "*.csv"))
right_files = glob.glob(os.path.join(path2, "*.csv"))

# Extract subject ID and file path
left_file_info = [(int(os.path.basename(f).split("-")[0]), f, "L") for f in left_files]
right_file_info = [
    (int(os.path.basename(f).split("-")[0]), f, "R") for f in right_files
]

# Combine and sort files by subject ID
file_info = left_file_info + right_file_info
file_info.sort(key=itemgetter(0))

# Group files by subject ID
grouped_files = {k: list(g) for k, g in groupby(file_info, key=itemgetter(0))}

# Initialize storage
data_intotal, target_intotal = [], []
data_left, target_left = [], []
data_right, target_right = [], []

# Process each subject's files
for subject_id, files in grouped_files.items():
    subject_data, subject_target = [], []
    subject_data_L, subject_target_L = [], []
    subject_data_R, subject_target_R = [], []

    for _, f, hand in files:
        df = pd.read_csv(f)
        df = df.drop(["t"], axis=1)
        target = df.pop("anno")

        # Save general subject data
        subject_data.append(df.values)
        subject_target.append(target.values)

        # Separate left and right hand data
        if hand == "L":
            subject_data_L.append(df.values)
            subject_target_L.append(target.values)
        else:
            subject_data_R.append(df.values)
            subject_target_R.append(target.values)

    # Concatenate per subject
    subject_data = np.concatenate(subject_data, axis=0)
    subject_target = np.concatenate(subject_target, axis=0)

    data_intotal.append(subject_data)
    target_intotal.append(subject_target)

    # Process left hand data
    if subject_data_L:
        subject_data_L = np.concatenate(subject_data_L, axis=0)
        subject_target_L = np.concatenate(subject_target_L, axis=0)
        data_left.append(subject_data_L)
        target_left.append(subject_target_L)

    # Process right hand data
    if subject_data_R:
        subject_data_R = np.concatenate(subject_data_R, axis=0)
        subject_target_R = np.concatenate(subject_target_R, axis=0)
        data_right.append(subject_data_R)
        target_right.append(subject_target_R)

# Verify label distribution
for i in range(len(target_intotal)):
    print(f"Subject {i+1}: {Counter(target_intotal[i])}")

# Ensure directory exists
os.makedirs("dataset/DX/DX-II", exist_ok=True)

# Save combined dataset
with open("dataset/DX/DX-II/DX_II_X.pkl", "wb") as f:
    pkl.dump(data_intotal, f)
with open("dataset/DX/DX-II/DX_II_Y.pkl", "wb") as f:
    pkl.dump(target_intotal, f)

# Save left hand dataset
with open("dataset/DX/DX-II/X_L.pkl", "wb") as f:
    pkl.dump(data_left, f)
with open("dataset/DX/DX-II/Y_L.pkl", "wb") as f:
    pkl.dump(target_left, f)

# Save right hand dataset
with open("dataset/DX/DX-II/X_R.pkl", "wb") as f:
    pkl.dump(data_right, f)
with open("dataset/DX/DX-II/Y_R.pkl", "wb") as f:
    pkl.dump(target_right, f)

print("DX-II data processing complete. Files saved.")
