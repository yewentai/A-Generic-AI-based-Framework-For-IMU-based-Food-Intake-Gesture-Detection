#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Checkpoint Utility Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-03
Description : This script provides utility functions for saving and loading model
              checkpoints during training, including tracking the best model state.
===============================================================================
"""

import os
import torch


def save_best_model(model, fold, current_metric, best_metric, checkpoint_dir, mode="max"):
    """
    Save the best model for a given fold based on a performance metric.

    Parameters:
        model (torch.nn.Module): The model to save.
        fold (int): Current fold number (e.g., 1, 2, ...).
        current_metric (float): The evaluation metric value from the current epoch.
        best_metric (float): The best metric value recorded so far.
        checkpoint_dir (str): Directory where the best model will be saved.
        mode (str): 'max' if higher metric values are better (e.g., accuracy),
                    'min' if lower metric values are better (e.g., loss).

    Returns:
        float: The updated best metric value.
    """
    # Determine if the current metric is better than the best metric so far
    is_better = (current_metric > best_metric) if mode == "max" else (current_metric < best_metric)

    if is_better:
        best_metric = current_metric
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    return best_metric


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

    print(f"Loaded checkpoint from fold {fold}, epoch {epoch} with F1 score: {f1_score:.4f}")
    return model, optimizer, epoch, f1_score


def save_generator(generator, optimizer_G, epoch, save_dir):
    """
    Save the generator model and optimizer state.

    Args:
        generator (nn.Module): The generator model.
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save the checkpoint.
    """
    gen_path = os.path.join(save_dir, f"generator_epoch_{epoch+1}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer_G.state_dict(),
        },
        gen_path,
    )


def save_discriminator(discriminator, optimizer_D, epoch, save_dir):
    """
    Save the discriminator model and optimizer state.

    Args:
        discriminator (nn.Module): The discriminator model.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save the checkpoint.
    """
    disc_path = os.path.join(save_dir, f"discriminator_epoch_{epoch+1}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": optimizer_D.state_dict(),
        },
        disc_path,
    )
