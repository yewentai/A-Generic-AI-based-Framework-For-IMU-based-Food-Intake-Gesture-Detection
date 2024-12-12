import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from collections import defaultdict

# Import the MultiStageTCN model
from model_tcn import MultiStageTCN
from utils import IMUDataset, segment_f1

def train_model(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_x, batch_y in train_loader:
        # Prepare input: [batch_size, num_channels, sequence_length]
        batch_x = batch_x.permute(0, 2, 1).to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)
        
        # Ensure output matches label dimensions
        outputs = outputs[:, :, -batch_y.shape[1]:].squeeze(-1)

        # Compute loss
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect predictions and labels
        predictions = (outputs > 0.5).float().cpu().numpy()
        all_predictions.append(predictions)
        all_labels.append(batch_y.cpu().numpy())
    
    # Calculate segment_f1
    segment_metrics = segment_f1(all_predictions, all_labels)

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Training - Loss: {avg_loss:.4f}, F1 (Segment): {segment_metrics}")

    return metrics

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Prepare input: [batch_size, num_channels, sequence_length]
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            
            # Ensure output matches label dimensions
            outputs = outputs[:, :, -batch_y.shape[1]:].squeeze(-1)

            # Compute loss
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()

            # Collect predictions and labels
            predictions = (outputs > 0.5).float().cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(batch_y.cpu().numpy())
    
    # Calculate segment_f1
    segment_metrics = segment_f1(all_predictions, all_labels)
    avg_loss = running_loss / len(test_loader)

    print(f"Testing - Loss: {avg_loss:.4f}, F1 (Segment): {segment_metrics}")

    return metrics

def loso_cross_validation(X, Y, config):
    num_subjects = len(X)
    all_metrics = defaultdict(list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_subject in range(num_subjects):
        print(f"\nTesting on subject {test_subject + 1}/{num_subjects}")

        # Prepare training and testing data
        train_X = [x for i, x in enumerate(X) if i != test_subject]
        train_Y = [y for i, y in enumerate(Y) if i != test_subject]
        test_X = [X[test_subject]]
        test_Y = [Y[test_subject]]

        # Create datasets and dataloaders
        train_dataset = IMUDataset(
            train_X,
            train_Y,
            sequence_length=config["sequence_length"]
        )
        test_dataset = IMUDataset(
            test_X,
            test_Y,
            sequence_length=config["sequence_length"]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False
        )

        # Initialize the model
        model = MultiStageTCN(
            input_channels=6,  # Assuming 6 input channels (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
            num_channels_per_stage=config["num_channels_per_stage"],
            kernel_size=config.get("kernel_size", 3),
            dropout=config.get("dropout", 0.2)
        ).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])

        # Training loop
        best_f1 = 0
        best_metrics = None

        for epoch in range(config["num_epochs"]):
            train_metrics = train_model(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                config["num_epochs"],
            )
            test_metrics = evaluate_model(model, test_loader, criterion, device)

            # Save the best model
            if test_metrics["f1"] > best_f1:
                best_f1 = test_metrics["f1"]
                best_metrics = test_metrics
                torch.save(model.state_dict(), f"best_model_subject_{test_subject}.pth")

        # Record the best results
        for metric, value in best_metrics.items():
            all_metrics[metric].append(value)

        print(f"\nBest metrics for subject {test_subject}:")
        for metric, value in best_metrics.items():
            print(f"{metric}: {value:.4f}")

    # Print overall results
    print("\nOverall LOSO Cross-validation Results:")
    for metric, values in all_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

    return all_metrics

# Load data
X_path = "./dataset/pkl_data/DX_I_X.pkl"
Y_path = "./dataset/pkl_data/DX_I_Y.pkl"
with open(X_path, "rb") as f:
    X = pickle.load(f)
with open(Y_path, "rb") as f:
    Y = pickle.load(f)

# Configuration parameters
config = {
    "sequence_length": 128,
    "downsample_factor": 4,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "num_channels_per_stage": [
        [16, 32],  # Stage 1
        [32, 64],  # Stage 2
        [64, 128], # Stage 3
    ],
    "kernel_size": 3,
    "dropout": 0.2
}

# Run LOSO cross validation
metrics = loso_cross_validation(X, Y, config)