import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
from models.model_tcn_mha import TCNMHA, TCNMHA_Loss
from augmentation import augment_orientation
from datasets import IMUDataset, segment_confusion_matrix, post_process_predictions
import logging

# Configuration
config = {
    "num_layers": 9,
    "num_classes": 3,
    "num_heads": 8,
    "input_dim": 6,
    "num_filters": 64,
    "kernel_size": 3,
    "dropout": 0.3,
    "lambda_coef": 0.15,
    "tau": 4,
    "learning_rate": 0.0005,
    "sampling_frequency": 16,
    "window_length": 60,
    "window_size": 16 * 60,  # 16 Hz * 60 s = 960
    "debug_plot": False,
    "X_L_path": "./dataset/FD/FD-I/X_L.pkl",
    "Y_L_path": "./dataset/FD/FD-I/Y_L.pkl",
    "X_R_path": "./dataset/FD/FD-I/X_R.pkl",
    "Y_R_path": "./dataset/FD/FD-I/Y_R.pkl",
    "num_epochs": 10,
    "batch_size": 1,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(path):
    with open(path, "rb") as f:
        return np.array(pickle.load(f), dtype=object)


def segment_data(data, labels, window_size):
    segmented_data = []
    segmented_labels = []
    for d, l in zip(data, labels):
        for i in range(0, len(d) - window_size + 1, window_size):
            segmented_data.append(d[i : i + window_size])
            segmented_labels.append(l[i : i + window_size])
    return np.array(segmented_data), np.array(segmented_labels)


def train_model(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        training_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = augment_orientation(batch_x)
            batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = TCNMHA_Loss(outputs, batch_y, config["lambda_coef"])
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        avg_train_loss = training_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}")


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
            outputs = model(batch_x)
            final_output = outputs[-1]
            probabilities = F.softmax(final_output, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            all_predictions.extend(predicted_classes.view(-1).cpu().numpy())
            all_labels.extend(batch_y.view(-1).cpu().numpy())
    return np.array(all_predictions), np.array(all_labels)


def main():
    device = torch.device(config["device"])
    logging.info(f"Using device: {device}")

    # Load and segment data
    X_L = load_data(config["X_L_path"])
    Y_L = load_data(config["Y_L_path"])
    X_R = load_data(config["X_R_path"])
    Y_R = load_data(config["Y_R_path"])
    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)
    X, Y = segment_data(X, Y, config["window_size"])

    # Create dataset and data loaders
    full_dataset = IMUDataset(X, Y)
    test_folds = [list(range(i * 10, (i + 1) * 10)) for i in range(7)]
    test_folds[-1] = list(range(60, 68))  # Adjust last fold

    training_statistics, testing_statistics = [], []
    loso_f1_scores = []

    # Generate a timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_stats_file = f"result/training_stats_tcnmha_{timestamp}.npy"
    testing_stats_file = f"result/testing_stats_tcnmha_{timestamp}.npy"

    for fold, test_subjects in enumerate(tqdm(test_folds, desc="7-Fold")):
        train_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject not in test_subjects
        ]
        test_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject in test_subjects
        ]
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        model = TCNMHA(
            input_dim=config["input_dim"],
            hidden_dim=config["num_filters"],
            num_classes=config["num_classes"],
            num_heads=config["num_heads"],
            d_model=128,
            kernel_size=config["kernel_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        train_model(model, train_loader, optimizer, device, config["num_epochs"])
        predictions, labels = evaluate_model(model, test_loader, device)
        predictions = post_process_predictions(predictions)

        # Calculate confusion matrix and F1 scores
        fn_25, fp_25, tp_25 = segment_confusion_matrix(
            predictions, labels, 0.25, config["debug_plot"]
        )
        fn_50, fp_50, tp_50 = segment_confusion_matrix(
            predictions, labels, 0.5, config["debug_plot"]
        )
        fn_75, fp_75, tp_75 = segment_confusion_matrix(
            predictions, labels, 0.75, config["debug_plot"]
        )
        f1_segment_25 = 2 * tp_25 / (2 * tp_25 + fp_25 + fn_25) if tp_25 > 0 else 0
        f1_segment_50 = 2 * tp_50 / (2 * tp_50 + fp_50 + fn_50) if tp_50 > 0 else 0
        f1_segment_75 = 2 * tp_75 / (2 * tp_75 + fp_75 + fn_75) if tp_75 > 0 else 0

        testing_statistics.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "f1_segment_25": f1_segment_25,
                "f1_segment_50": f1_segment_50,
                "f1_segment_75": f1_segment_75,
            }
        )
        loso_f1_scores.append([f1_segment_25, f1_segment_50, f1_segment_75])

    # Save results with timestamped file names
    np.save(training_stats_file, training_statistics)
    np.save(testing_stats_file, testing_statistics)
    logging.info(f"Training statistics saved to {training_stats_file}")
    logging.info(f"Testing statistics saved to {testing_stats_file}")


if __name__ == "__main__":
    main()
