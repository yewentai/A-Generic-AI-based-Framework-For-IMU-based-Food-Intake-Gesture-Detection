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

# Load all files
path1 = r'dataset/DX-I/Raw data and Annotation/Left hand'
path2 = r'dataset/DX-I/Raw data and Annotation/Right hand'
files = glob.glob(os.path.join(path1, '*.csv')) + glob.glob(os.path.join(path2, '*.csv'))

# Extract subject ID from each file name
file_info = [(int(os.path.basename(f).split('-')[0]), f) for f in files]

# Sort files by subject ID
file_info.sort(key=itemgetter(0))

# Group files by subject ID
grouped_files = {k: list(map(itemgetter(1), g)) for k, g in groupby(file_info, key=itemgetter(0))}

data_intotal = []  # list to save 6-axis IMU data
target_intotal = []  # list to save annotation

for subject_id, files in grouped_files.items():
    subject_data = []
    subject_target = []
    for f in files:
        df = pd.read_csv(f)
        df = df.drop(['t'], axis=1)
        target1 = df.pop('anno')
        subject_data.append(df.values)
        subject_target.append(target1.values)
    
    # Concatenate all data and targets for the same subject
    subject_data = np.concatenate(subject_data, axis=0)
    subject_target = np.concatenate(subject_target, axis=0)
    
    data_intotal.append(subject_data)
    target_intotal.append(subject_target)

# Verify the distribution of labels for each subject
for i in range(len(target_intotal)):
    print(f"Subject {i+1}: ", Counter(target_intotal[i]))

# Save the concatenated data to pickle files
with open('dataset/pkl_data/DX_I_X_raw.pkl', 'wb') as f:
    pkl.dump(data_intotal, f)
with open('dataset/pkl_data/DX_I_Y_raw.pkl', 'wb') as f:
    pkl.dump(target_intotal, f)



""""
Process the raw data and annotations for DX-II dataset
"""

# Load all files
path1 = r'dataset/DX-II/Raw data and Annotation/Left hand'
path2 = r'dataset/DX-II/Raw data and Annotation/Right hand'
files = glob.glob(os.path.join(path1, '*.csv')) + glob.glob(os.path.join(path2, '*.csv'))

# Extract subject ID from each file name
file_info = [(int(os.path.basename(f).split('-')[0]), f) for f in files]

# Sort files by subject ID
file_info.sort(key=itemgetter(0))

# Group files by subject ID
grouped_files = {k: list(map(itemgetter(1), g)) for k, g in groupby(file_info, key=itemgetter(0))}
data_intotal = []  # list to save 6-axis IMU data
target_intotal = []  # list to save annotation

for subject_id, files in grouped_files.items():
    subject_data = []
    subject_target = []
    for f in files:
        df = pd.read_csv(f)
        df = df.drop(['t'], axis=1)
        target1 = df.pop('anno')
        subject_data.append(df.values)
        subject_target.append(target1.values)
    
    # Concatenate all data and targets for the same subject
    subject_data = np.concatenate(subject_data, axis=0)
    subject_target = np.concatenate(subject_target, axis=0)
    
    data_intotal.append(subject_data)
    target_intotal.append(subject_target)

# Verify the distribution of labels for each subject
for i in range(len(target_intotal)):
    print(f"Subject {i+1}: ", Counter(target_intotal[i]))

# Save the concatenated data to pickle files
with open('dataset/pkl_data/DX_II_X_raw.pkl', 'wb') as f:
    pkl.dump(data_intotal, f)
with open('dataset/pkl_data/DX_II_Y_raw.pkl', 'wb') as f:
    pkl.dump(target_intotal, f)