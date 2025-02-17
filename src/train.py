# -*- coding: utf-8 -*-
"""Simplified training script without K-Fold validation."""

import os
import pickle
from datetime import date, datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from augmentation import augment_orientation
from datasets import IMUDataset, post_process_predictions, segment_confusion_matrix
from models.model_mstcn import MSTCN, MSTCN_Loss

# region Constants & Configuration
MODEL_NAME = "MSTCN"
DATASET_NAME = "FD-I"
TODAY = date.today().strftime("%Y%m%d")

NUM_LAYERS = 9
NUM_CLASSES = 3
NUM_HEADS = 8
INPUT_DIM = 6
NUM_FILTERS = 64
KERNEL_SIZE = 3
DROPOUT = 0.3
LAMBDA_COEF = 0.15
LEARNING_RATE = 0.0005
SAMPLING_FREQUENCY = 16
WINDOW_LENGTH = 60  # seconds
WINDOW_SIZE = SAMPLING_FREQUENCY * WINDOW_LENGTH
DEBUG_PLOT = False
BATCH_SIZE = 64
NUM_EPOCHS = 10
TRAIN_RATIO = 0.8

PATHS = {
    "X_L": "./dataset/FD/FD-I/X_L.pkl",
    "Y_L": "./dataset/FD/FD-I/Y_L.pkl",
    "X_R": "./dataset/FD/FD-I/X_R.pkl",
    "Y_R": "./dataset/FD/FD-I/Y_R.pkl",
    "results": "result",
    "checkpoints": "checkpoints",
}
# endregion


def setup(rank: int, world_size: int) -> None:
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup() -> None:
    """Clean up distributed training resources."""
    dist.destroy_process_group()


def segment_data(
    data: np.ndarray, labels: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Segment time series data into fixed windows."""
    segmented_data, segmented_labels = [], []
    for d, l in zip(data, labels):
        for i in range(0, len(d) - window_size + 1, window_size):
            segmented_data.append(d[i : i + window_size])
            segmented_labels.append(l[i : i + window_size])
    return np.array(segmented_data), np.array(segmented_labels)


def get_checkpoint_path(epoch: int = None, best: bool = False) -> str:
    """Generate standardized checkpoint paths."""
    base_dir = os.path.join(
        PATHS["checkpoints"], f"{MODEL_NAME}_{DATASET_NAME}_{TODAY}"
    )
    os.makedirs(base_dir, exist_ok=True)

    if best:
        return os.path.join(base_dir, "best_model.pt")
    if epoch:
        return os.path.join(base_dir, f"epoch_{epoch}.pt")
    return base_dir


def save_checkpoint(model, optimizer, scaler, epoch: int, is_best: bool = False):
    """Save training state with proper naming."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    path = (
        get_checkpoint_path(best=True) if is_best else get_checkpoint_path(epoch=epoch)
    )
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}") if dist.get_rank() == 0 else None


def load_checkpoint(rank: int):
    """Load latest checkpoint for resuming training."""
    try:
        checkpoint_dir = get_checkpoint_path()
        all_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        epochs = [
            int(f.split("_")[1].split(".")[0]) for f in all_checkpoints if "epoch" in f
        ]
        latest_epoch = max(epochs) if epochs else None

        if latest_epoch:
            path = get_checkpoint_path(epoch=latest_epoch)
            checkpoint = torch.load(path, map_location=f"cuda:{rank}")
            print(f"Resuming training from checkpoint: {path}") if rank == 0 else None
            return checkpoint, latest_epoch
    except Exception as e:
        print(f"No checkpoints found: {e}") if rank == 0 else None
    return None, 0


def main(rank: int, world_size: int) -> None:
    """Main training loop without K-Fold."""
    setup(rank, world_size)

    # region Data Loading & Preparation
    if rank == 0:
        datasets = {}
        for key in ["X_L", "Y_L", "X_R", "Y_R"]:
            with open(PATHS[key], "rb") as f:
                datasets[key] = np.array(pickle.load(f), dtype=object)

        X = np.concatenate([datasets["X_L"], datasets["X_R"]], axis=0)
        Y = np.concatenate([datasets["Y_L"], datasets["Y_R"]], axis=0)
        X, Y = segment_data(X, Y, WINDOW_SIZE)
        full_dataset = IMUDataset(X, Y)
    else:
        full_dataset = None

    full_dataset = dist.broadcast_object(full_dataset, src=0)

    # Split dataset
    train_size = int(TRAIN_RATIO * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    # endregion

    # region Model & Optimizer Setup
    model = MSTCN(
        input_dim=INPUT_DIM,
        hidden_dim=NUM_FILTERS,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        d_model=128,
        kernel_size=KERNEL_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * world_size)
    scaler = torch.amp.GradScaler()
    # endregion

    # region Checkpoint Loading
    checkpoint, start_epoch = load_checkpoint(rank)
    if checkpoint:
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    # endregion

    # region Training Setup
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    best_loss = float("inf")
    training_stats = []
    # endregion

    # region Training Loop
    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = augment_orientation(batch_x)
            batch_x = batch_x.permute(0, 2, 1).to(rank)
            batch_y = batch_y.to(rank)

            optimizer.zero_grad()
            with torch.amp.autocast():
                outputs = model(batch_x)
                loss = MSTCN_Loss(outputs, batch_y, LAMBDA_COEF)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Save checkpoints and statistics
        if rank == 0:
            training_stats.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "epoch": epoch,
                    "train_loss": avg_loss,
                }
            )

            # Save regular checkpoint
            save_checkpoint(model, optimizer, scaler, epoch)

            # Update best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, scaler, epoch, is_best=True)
    # endregion

    # region Final Evaluation
    if rank == 0:
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.permute(0, 2, 1).to(rank)
                outputs = model(batch_x)
                probs = F.softmax(outputs[-1], dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        processed_preds = post_process_predictions(np.array(all_preds))
        fn, fp, tp = segment_confusion_matrix(
            processed_preds, np.array(all_labels), threshold=0.5, debug_plot=DEBUG_PLOT
        )

        f1_segment = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0

        # Save results
        results_prefix = f"{MODEL_NAME}_{DATASET_NAME}_{TODAY}"
        np.save(
            os.path.join(PATHS["results"], f"training_stats_{results_prefix}.npy"),
            training_stats,
        )

        testing_stats = {
            "timestamp": datetime.now().isoformat(),
            "f1_segment": f1_segment,
            "fn": fn,
            "fp": fp,
            "tp": tp,
        }
        np.save(
            os.path.join(PATHS["results"], f"testing_stats_{results_prefix}.npy"),
            testing_stats,
        )

        # Save final model
        final_model_path = os.path.join(PATHS["results"], f"{results_prefix}_final.pt")
        torch.save(model.module.state_dict(), final_model_path)
    # endregion

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
