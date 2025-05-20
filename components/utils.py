"""
================================================================================================
Loss Functions and Temporal Smoothing Utilities
------------------------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-21
Description : This script provides utility functions for computing temporal smoothing losses
              for sequence models such as MSTCN. It supports various smoothing strategies
              (e.g., MSE, L1, Huber, JS, TV, etc.) and includes a deep-supervised loss
              wrapper for multi-stage predictions. These functions are designed to enhance
              model performance by penalizing abrupt temporal changes in predictions.
================================================================================================
"""

import torch
import torch.nn.functional as F


def loss_fn(outputs, targets, smoothing="L1", max_diff=16.0):
    """
    Compute loss: cross-entropy + various temporal smoothing penalties.

    Args:
        outputs (Tensor): logits of shape [B, C, L].
        targets (Tensor): labels of shape [B, L], long, with ignore_index=-100.
        smoothing (str): one of ['MSE','L1','HUBER','JS','TV','SEC_DIFF','EMD'].
        max_diff (float): clamp threshold for log-prob differences (only for diff-based).
    Returns:
        ce_loss (Tensor): the CrossEntropyLoss.
        smooth_loss (Tensor): the chosen smoothing loss.
    """
    # --- Cross‐Entropy Part ---
    # outputs: [B, C, L], targets: [B, L]
    logits = outputs  # [B, C, L]
    labels = targets.long()  # [B, L]
    # flatten logits to [B*L, C] and labels to [B*L]
    logits_flat = logits.permute(0, 2, 1).flatten(0, 1)  # swap → [B, L, C], then flatten two dims
    labels_flat = labels.flatten()  # [B*L]
    # one‐liner cross‐entropy
    ce_loss = F.cross_entropy(logits_flat, labels_flat)

    # --- Smoothing Part ---
    # Precompute log-probs and shifted versions
    logp = F.log_softmax(outputs, dim=1)  # [B,C,L]
    log_prev = logp[:, :, :-1].detach()  # stop grad beyond prev
    log_next = logp[:, :, 1:]
    diff = log_next - log_prev  # [B,C,L-1]

    # Select smoothing
    if smoothing == "MSE":
        smooth_loss = diff.clamp(-max_diff, max_diff).pow(2).mean()
        coefficient = 0.5

    elif smoothing == "L1":
        smooth_loss = diff.clamp(-max_diff, max_diff).abs().mean()
        coefficient = 0.3

    elif smoothing == "HUBER":
        # smooth_l1: L2 for small, L1 for large
        smooth_loss = F.smooth_l1_loss(log_next, log_prev, reduction="mean")
        coefficient = 0.5

    elif smoothing == "JS":
        # Jensen–Shannon divergence
        p_next = log_next.exp()
        p_prev = log_prev.exp()
        m = 0.5 * (p_next + p_prev)
        # KL(P||M) + KL(Q||M)
        smooth_loss = 0.5 * (
            F.kl_div(log_next, m.detach(), reduction="batchmean")
            + F.kl_div(log_prev, m.detach(), reduction="batchmean")
        )
        coefficient = 1.0

    elif smoothing == "TV":
        # Total Variation: sum over classes, mean over batch+time
        smooth_loss = diff.abs().sum(dim=1).mean()
        coefficient = 0.2

    elif smoothing == "SEC_DIFF":
        # Second‐order difference: penalize change of slope
        d1 = diff  # [B,C,L-1]
        d2 = d1[:, :, 1:] - d1[:, :, :-1]  # [B,C,L-2]
        smooth_loss = d2.pow(2).mean()
        coefficient = 0.5

    elif smoothing == "EMD":
        # 1D EMD via L1 of CDF differences
        p_next = log_next.exp()
        p_prev = log_prev.exp()
        cdf_next = torch.cumsum(p_next, dim=1)
        cdf_prev = torch.cumsum(p_prev, dim=1)
        smooth_loss = (cdf_next - cdf_prev).abs().mean()
        coefficient = 15.0

    elif smoothing == "None":
        # No smoothing
        smooth_loss = torch.tensor(0.0, device=logits.device)
        coefficient = 0.0

    else:
        raise ValueError(f"Unknown smoothing type '{smoothing}'")

    return ce_loss, coefficient * smooth_loss


def MSTCN_Loss(logits_list, targets, smoothing="MSE", max_diff=16.0):
    """
    Compute deep‐supervised loss for MSTCN.

    Args:
        logits_list   (List[Tensor]): list of stage‐outputs, each [B,C,L].
        targets       (Tensor): [B,L] long with ignore_index=-100.
        smoothing     (str): which smoothing penalty to apply.
        max_diff      (float): clamp threshold for diff‐based.

    Returns:
        Tuple(Tensor, Tensor): total_ce_loss, total_smooth_loss
    """
    total_ce = 0.0
    total_smooth = 0.0

    for logits in logits_list:
        ce, smooth = loss_fn(outputs=logits, targets=targets, smoothing=smoothing, max_diff=max_diff)
        total_ce += ce
        total_smooth += smooth

    return total_ce, total_smooth
