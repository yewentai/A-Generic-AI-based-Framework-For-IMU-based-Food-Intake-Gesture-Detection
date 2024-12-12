# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from model_cnnlstm import CNN_LSTM

# %%
# Load .pkl data
# Paths to .pkl files
X_path = "./dataset/pkl_data/DX_I_X.pkl"
Y_path = "./dataset/pkl_data/DX_I_Y.pkl"
with open(X_path, "rb") as f:
    X = pickle.load(f)  # List of numpy arrays
with open(Y_path, "rb") as f:
    Y = pickle.load(f)  # List of numpy arrays

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

# Prepare dataset and dataloader
dataset = IMUDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
def train_model(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.permute(0, 2, 1).to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        outputs = outputs.squeeze(-1)

        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect predictions and labels for calculating metrics
        predictions = (outputs > 0.5).float().cpu().numpy()
        all_predictions.extend(predictions.flatten())
        all_labels.extend(batch_y.cpu().numpy().flatten())

    # Calculating training metrics
    metrics = calculate_metrics(all_labels, all_predictions)

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(
        f"Training - Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )

    return metrics


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, batch_y)
            running_loss += loss.item()

            predictions = (outputs > 0.5).float().cpu().numpy()
            all_predictions.extend(predictions.flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())

    metrics = calculate_metrics(all_labels, all_predictions)
    avg_loss = running_loss / len(test_loader)

    print(
        f"Testing - Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )

    return metrics


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# LOSO cross validation main function
def loso_cross_validation(X, Y, config):
    num_subjects = len(X)
    all_metrics = defaultdict(list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_subject in range(num_subjects):
        print(f"\nTesting on subject {test_subject + 1}/{num_subjects}")

        # Preparing training and testing data
        train_X = [x for i, x in enumerate(X) if i != test_subject]
        train_Y = [y for i, y in enumerate(Y) if i != test_subject]
        test_X = [X[test_subject]]
        test_Y = [Y[test_subject]]

        # Creating a dataset and data loader
        train_dataset = IMUDataset(
            train_X,
            train_Y,
            sequence_length=config["sequence_length"],
            downsample_factor=config["downsample_factor"],
        )
        test_dataset = IMUDataset(
            test_X,
            test_Y,
            sequence_length=config["sequence_length"],
            downsample_factor=config["downsample_factor"],
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False
        )

        # Initialize the model and optimizer
        model = CNN_LSTM().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])

        # Training the model
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
                torch.save(model.state_dict(), f"models/best_cnnlstm_model_subject_{test_subject}.pth")

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

# %%
# Configuration parameters
config = {
    "sequence_length": 128,
    "downsample_factor": 4,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "num_epochs": 20,
}

# Run LOSO cross validation
metrics = loso_cross_validation(X, Y, config)
