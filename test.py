import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
from model_mstcn import MSTCN
from utils import IMUDataset, segment_f1_drinking

# Hyperparameters
num_stages = 2
num_layers = 9
num_classes = 2
input_dim = 6
num_filters = 128
kernel_size = 3
dropout = 0.3
lambda_coef = 0.15
tau = 4

# Load data
X_path = "./dataset/pkl_data/DX_I_X.pkl"
Y_path = "./dataset/pkl_data/DX_I_Y.pkl"
with open(X_path, "rb") as f:
    X = pickle.load(f)
with open(Y_path, "rb") as f:
    Y = pickle.load(f)

# Prepare datasets
full_dataset = IMUDataset(X, Y)
print(f"Dataset size: {len(full_dataset)}")
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing model
model = MSTCN(num_stages=num_stages, num_layers=num_layers, 
                      num_classes=num_classes, input_dim=input_dim, 
                      num_filters=num_filters, kernel_size=kernel_size, 
                      dropout=dropout)
model.load_state_dict(torch.load("models/mstcn_model.pth", weights_only=True))
model.to(device)
model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
        
        # Get the outputs from the model
        outputs = model(batch_x)

        # Use the output from the last stage
        final_output = outputs[-1]  # Assuming we want to use the last stage's output
        
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(final_output, dim=1)  # Shape: [batch_size, num_classes, seq_len]

        # Get the predicted class (class with the highest probability)
        predicted_classes = torch.argmax(probabilities, dim=1)  # Shape: [batch_size, seq_len]

        # Move predictions and true labels to CPU for evaluation and flatten the arrays
        all_predictions.append(predicted_classes.cpu().numpy().flatten())
        all_labels.append(batch_y.cpu().numpy().flatten().astype(int))

# Calculate the precision, recall, and F1 score
fp = 0
fn = 0
tp = 0
for i in range(len(all_predictions)):
    fp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 0))
    fn += np.sum((all_predictions[i] == 0) & (all_labels[i] == 1))
    tp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"Test - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

segment_metrics = segment_f1_drinking(all_predictions, all_labels)
print(f"Test - F1 (Segment): {segment_metrics}")

# Test model with augmentation
model.load_state_dict(torch.load("models/mstcn_model_aug.pth", weights_only=True))
model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
        
        # Get the outputs from the model
        outputs = model(batch_x)

        # Use the output from the last stage
        final_output = outputs[-1]  # Assuming we want to use the last stage's output
        
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(final_output, dim=1)  # Shape: [batch_size, num_classes, seq_len]

        # Get the predicted class (class with the highest probability)
        predicted_classes = torch.argmax(probabilities, dim=1)  # Shape: [batch_size, seq_len]

        # Move predictions and true labels to CPU for evaluation and flatten the arrays
        all_predictions.append(predicted_classes.cpu().numpy().flatten())
        all_labels.append(batch_y.cpu().numpy().flatten().astype(int))

# Calculate the precision, recall, and F1 score
fp = 0
fn = 0
tp = 0
for i in range(len(all_predictions)):
    fp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 0))
    fn += np.sum((all_predictions[i] == 0) & (all_labels[i] == 1))
    tp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"Test (Augmented) - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

segment_metrics = segment_f1_drinking(all_predictions, all_labels)
print(f"Test (Augmented) - F1 (Segment): {segment_metrics}")