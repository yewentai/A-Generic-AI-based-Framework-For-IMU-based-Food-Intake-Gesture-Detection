# Add this code to your imports at the top of the file
import torch
import os


def save_checkpoint(model, optimizer, epoch, fold, f1_score, is_best=False):
    """
    Save model checkpoint to file

    Args:
        model: The PyTorch model to save
        optimizer: The optimizer used for training
        epoch: Current epoch number
        fold: Current fold number
        f1_score: F1 score on validation/test set
        is_best: Boolean indicating if this is the best model so far
    """
    os.makedirs("checkpoint", exist_ok=True)

    checkpoint = {
        "fold": fold,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "f1_score": f1_score,
    }

    # Save latest checkpoint
    checkpoint_path = f"checkpoint/fold{fold}_epoch{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)

    # If best model, save a copy
    if is_best:
        best_path = f"checkpoint/fold{fold}_best.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model for fold {fold} with F1 score: {f1_score:.4f}")


# Function to load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model from checkpoint

    Args:
        model: The PyTorch model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file

    Returns:
        model: The model with loaded weights
        optimizer: The optimizer with loaded state
        epoch: The epoch number when checkpoint was saved
        f1_score: The F1 score when checkpoint was saved
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    f1_score = checkpoint["f1_score"]
    fold = checkpoint["fold"]

    print(
        f"Loaded checkpoint from fold {fold}, epoch {epoch} with F1 score: {f1_score:.4f}"
    )
    return model, optimizer, epoch, f1_score
