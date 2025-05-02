"""
================================================================================================
Loss Functions and MATLAB-Compatible Conversion Utilities
------------------------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-02
Description : This script includes utility functions for converting nested Python structures
              into MATLAB-compatible formats and computing various temporal smoothing losses
              for sequence models such as MSTCN. It supports multiple smoothing strategies
              (e.g., MSE, L1, Huber, KL, JS, TV, etc.) and includes a deep-supervised loss
              wrapper for multi-stage predictions.
================================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_for_matlab(data, context=None):
    """Recursively convert Python dict to MATLAB-friendly format."""
    if isinstance(data, dict):
        converted = {}
        for key, value in data.items():
            # Ensure the key is a string.
            original_key = key
            if not isinstance(key, str):
                key = str(key)

            current_key = key
            current_context = context

            # Update context for nested structures (before key conversion!)
            if key == "metrics_segment":
                current_context = "metrics_segment_thresholds"
            elif context == "metrics_segment_thresholds":
                current_context = "metrics_segment_classes"
            elif key == "metrics_sample":
                current_context = "metrics_sample_metrics"
            elif key == "label_distribution":
                current_context = "label_distribution_values"

            # Handle key conversions based on the current context
            if isinstance(key, str):
                # If we're processing threshold keys (from "metrics_segment"),
                # convert them from (e.g., "0.1") to a valid field name (e.g., "t10").
                if context == "metrics_segment_thresholds":
                    try:
                        num = float(key)
                        num_int = int(round(num * 100))
                        current_key = f"t{num_int:02d}"  # Ensure two-digit format
                    except ValueError:
                        pass
                # For contexts where keys represent class labels or metrics,
                # prepend a letter to ensure the key is a valid MATLAB field name.
                elif current_context in {
                    "metrics_segment_classes",
                    "metrics_sample_metrics",
                    "label_distribution_values",
                }:
                    # If the key consists of digits only, convert it.
                    if key.isdigit():
                        current_key = f"c{key}"
            # If key conversion did not result in a valid string, fall back to the original string key.
            current_key = str(current_key)

            # Recurse into sub-structures
            converted_value = convert_for_matlab(value, context=current_context)
            converted[current_key] = converted_value

        return converted

    elif isinstance(data, list):
        return [convert_for_matlab(item, context=context) for item in data]
    else:
        return data


def loss_fn(outputs, targets, smoothing="MSE", max_diff=16.0):
    """
    Compute loss: cross-entropy + various temporal smoothing penalties.

    Args:
        outputs (Tensor): logits of shape [B, C, L].
        targets (Tensor): labels of shape [B, L], long, with ignore_index=-100.
        smoothing (str): one of ['MSE','L1','HUBER','KL','JS','TV','SEC_DIFF','EMD','SPECTRAL'].
        max_diff (float): clamp threshold for log-prob differences (only for diff-based).
    Returns:
        ce_loss (Tensor): the CrossEntropyLoss.
        smooth_loss (Tensor): the chosen smoothing loss.
    """
    # --- Cross‐Entropy Part ---
    targets = targets.long()
    ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    B, C, L = outputs.size()
    logits_flat = outputs.transpose(2, 1).contiguous().view(-1, C)
    targets_flat = targets.view(-1)
    ce_loss = ce_fn(logits_flat, targets_flat)

    # If seq_len == 1, skip smoothing
    if L <= 1:
        return ce_loss, torch.tensor(0.0, device=outputs.device)

    # Precompute log-probs and shifted versions
    logp = F.log_softmax(outputs, dim=1)  # [B,C,L]
    log_prev = logp[:, :, :-1].detach()  # stop grad beyond prev
    log_next = logp[:, :, 1:]
    diff = log_next - log_prev  # [B,C,L-1]

    # Select smoothing
    if smoothing == "MSE":
        smooth_loss = diff.clamp(-max_diff, max_diff).pow(2).mean()

    elif smoothing == "L1":
        smooth_loss = diff.clamp(-max_diff, max_diff).abs().mean()

    elif smoothing == "HUBER":
        # smooth_l1: L2 for small, L1 for large
        smooth_loss = F.smooth_l1_loss(log_next, log_prev, reduction="mean")

    elif smoothing == "KL":
        # KL( P_{t+1} || P_t )
        smooth_loss = F.kl_div(log_next, log_prev.exp(), reduction="batchmean")

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

    elif smoothing == "TV":
        # Total Variation: sum over classes, mean over batch+time
        smooth_loss = diff.abs().sum(dim=1).mean()

    elif smoothing == "SEC_DIFF":
        # Second‐order difference: penalize change of slope
        d1 = diff  # [B,C,L-1]
        d2 = d1[:, :, 1:] - d1[:, :, :-1]  # [B,C,L-2]
        smooth_loss = d2.pow(2).mean()

    elif smoothing == "EMD":
        # 1D EMD via L1 of CDF differences
        p_next = log_next.exp()
        p_prev = log_prev.exp()
        cdf_next = torch.cumsum(p_next, dim=1)
        cdf_prev = torch.cumsum(p_prev, dim=1)
        smooth_loss = (cdf_next - cdf_prev).abs().mean()

    elif smoothing == "SPECTRAL":
        # Spectral high‐freq penalty: mean power of non‐DC freqs
        fft = torch.fft.rfft(logp, dim=2)
        mag2 = fft[..., 1:].abs().pow(2)
        smooth_loss = mag2.mean()

    else:
        raise ValueError(f"Unknown smoothing type '{smoothing}'")

    return ce_loss, smooth_loss


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
