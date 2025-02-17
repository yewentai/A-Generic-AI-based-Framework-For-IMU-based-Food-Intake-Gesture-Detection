import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from augmentation import augment_orientation
from utils import IMUDataset, segment_confusion_matrix, post_process_predictions
from model_cnnlstm import CNN_LSTM
from utils import IMUDatasetDowmSample

# Hyperparameters and configuration
config = {
    "sequence_length": 128,
    "downsample_factor": 4,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "num_epochs": 20,
}

# Loading data
X_path = "./dataverse_files/pkl_data/pkl_data/DX_I_X.pkl"
Y_path = "./dataverse_files/pkl_data/pkl_data/DX_I_Y.pkl"
with open(X_path, "rb") as f:
    X = pickle.load(f)  # List of numpy arrays
with open(Y_path, "rb") as f:
    Y = pickle.load(f)  # List of numpy arrays

# Prepare dataset and dataloader
dataset = IMUDatasetDowmSample(X, Y)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving result
if not os.path.exists("result"):
    os.makedirs("result")

# File names for training and testing result
training_stats_file = "result/training_stats_mstcn.npy"
testing_stats_file = "result/testing_stats_mstcn.npy"

# Initialize empty lists to store result
training_statistics = []
testing_statistics = []

# LOSO cross-validation
unique_subjects = np.unique(dataset.subject_indices)
loso_f1_scores = []
