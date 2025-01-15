import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for saving models
save_path = "./checkpoints"
os.makedirs(save_path, exist_ok=True)

# Initialize k-fold cross validation
kfold = KFold(n_splits=7, shuffle=True, random_state=42)
fold_results = []

# Initialize lists to store results
all_train_losses = []
all_val_f1_scores = []

# Perform k-fold cross validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
    print(f"\nStarting fold {fold+1}/7")

    # Create data loaders for this fold
    train_loader = DataLoader(
        Subset(full_dataset, train_idx), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = TCNMHA(
        input_dim=input_dim,
        hidden_dim=64,
        num_classes=num_classes,
        num_heads=8,
        d_model=128,
        kernel_size=3,
        num_layers=9,
    ).to(device)

    criterion = TCNMHALoss(alpha=0.15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Training loop variables
    best_val_f1 = 0
    train_losses = []
    val_f1_scores = []
    patience = 10
    counter = 0

    # Training loop
    for epoch in range(100):  # Max epochs with early stopping
        # Training phase
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold+1}, Epoch {epoch+1}")

        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                predictions = torch.argmax(outputs, dim=-1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_f1 = segment_f1_multiclass(np.array(all_preds), np.array(all_labels))
        val_f1_scores.append(val_f1)

        print(
            f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val F1 = {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            counter = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                },
                f"{save_path}/tcn_mha_fold{fold+1}_best_{timestamp}.pth",
            )
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Store results for this fold
    fold_results.append(best_val_f1)
    all_train_losses.append(train_losses)
    all_val_f1_scores.append(val_f1_scores)

    print(f"Fold {fold+1} Best Validation F1: {best_val_f1:.4f}")

# Print final results
mean_f1 = np.mean(fold_results)
std_f1 = np.std(fold_results)
print(f"\nCross-validation results:")
print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

# Plot results
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 2, 1)
for fold in range(7):
    plt.plot(all_train_losses[fold], label=f"Fold {fold+1}")
plt.title("Training Loss by Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot validation F1 scores
plt.subplot(1, 2, 2)
for fold in range(7):
    plt.plot(all_val_f1_scores[fold], label=f"Fold {fold+1}")
plt.title("Validation F1 Score by Fold")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()

plt.tight_layout()
plt.savefig("cross_validation_results.png")
plt.close()
