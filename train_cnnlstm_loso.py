import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pickle
import csv
from datetime import datetime
from tqdm import tqdm
from augmentation import augment_orientation
from model_cnnlstm import CNN_LSTM
from utils import post_process_predictions

# Dataset class
class IMUDataset(Dataset):
    def __init__(self, X, Y, sequence_length=128, downsample_factor=4):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.subject_indices = []  # Record which subject each sample belongs to

        # Processing data for each session
        for subject_idx, (imu_data, labels) in enumerate(zip(X, Y)):
            # imu_data = self.normalize(imu_data)
            num_samples = len(labels)

            for i in range(0, num_samples, sequence_length):
                imu_segment = imu_data[i : i + sequence_length]
                label_segment = labels[i : i + sequence_length]

                if len(imu_segment) == sequence_length:
                    self.data.append(imu_segment)
                    downsampled_labels = label_segment[:: self.downsample_factor]
                    self.labels.append(downsampled_labels)
                    self.subject_indices.append(subject_idx)

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# Hyperparameters
sequence_length = 128
downsample_factor = 4
batch_size = 32
learning_rate = 1e-3
num_epochs = 20
criterion = nn.BCELoss()

# Load .pkl data
X_path = "./dataset/pkl_data/DX_I_X_mirrored.pkl"
Y_path = "./dataset/pkl_data/DX_I_Y_mirrored.pkl"
with open(X_path, "rb") as f:
    X = pickle.load(f)  # List of numpy arrays
with open(Y_path, "rb") as f:
    Y = pickle.load(f)  # List of numpy arrays

# Create the full dataset
full_dataset = IMUDataset(X, Y)

# LOSO cross-validation
unique_subjects = np.unique(full_dataset.subject_indices)
loso_f1_scores = []

# Open CSV files for writing
with open("result/training_log_dxi_mirrored_cnnlstm.csv", mode='w', newline='') as train_csvfile, \
     open("result/testing_log_dxi_mirrored_cnnlstm.csv", mode='w', newline='') as test_csvfile:

    train_csv_writer = csv.writer(train_csvfile)
    test_csv_writer = csv.writer(test_csvfile)

    # Write headers
    train_csv_writer.writerow(['Date', 'Time', 'Fold', 'Epoch', 'Training Loss'])
    test_csv_writer.writerow(['Date', 'Time', 'Fold', 'F1 Sample'])

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, test_subject in enumerate(tqdm(unique_subjects, desc="LOSO Folds", leave=True)):
        # Create train and test indices
        train_indices = [i for i, subject in enumerate(full_dataset.subject_indices) if subject != test_subject]
        test_indices = [i for i, subject in enumerate(full_dataset.subject_indices) if subject == test_subject]

        # Create Subset datasets
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

        # Model initialization
        model = CNN_LSTM().to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        # Training loop
        num_epochs = 20
        for epoch in tqdm(range(num_epochs), desc=f"Training Fold {fold + 1}", leave=False):
            model.train()
            training_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # Data augmentation
                batch_x = augment_orientation(batch_x)
                batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)

                outputs = model(batch_x)
                outputs = outputs.squeeze(-1)

                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

            avg_train_loss = training_loss / len(train_loader)

            # Save training data into CSV
            now = datetime.now()
            train_csv_writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), 
                                    fold + 1, epoch + 1, avg_train_loss])   
            
        # Testing
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
                outputs = model(batch_x)
                final_output = outputs[-1]
                probabilities = F.softmax(final_output, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predicted_classes.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        # Post-processing predictions
        all_predictions = post_process_predictions(all_predictions)

        # Calculate metrics
        fp, fn, tp = 0, 0, 0
        for i in range(len(all_predictions)):
            fp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 0))
            fn += np.sum((all_predictions[i] == 0) & (all_labels[i] == 1))
            tp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_sample = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Save the F1 scores after testing
        now = datetime.now()
        test_csv_writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), 
                                  fold + 1, f1_sample])