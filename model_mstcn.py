import torch
import torch.nn as nn
import torch.nn.functional as F


# Dilated Residual Layer
class DilatedResidualLayer(nn.Module):
    def __init__(self, num_filters, dilation, kernel_size=3, dropout=0.3):
        super(DilatedResidualLayer, self).__init__()
        # Dilated convolution
        self.conv_dilated = nn.Conv1d(
            num_filters, num_filters, kernel_size, padding=dilation, dilation=dilation
        )
        # 1x1 convolution
        self.conv_1x1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Dilated convolution
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.dropout(out)
        # 1x1 convolution
        out = self.conv_1x1(out)
        # Residual connection
        out = out + x
        return out


# Single Stage TCN module
class SSTCN(nn.Module):

    def __init__(
        self,
        num_layers,
        in_channels,
        num_classes,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
        num_heads=8,
    ):
        super(SSTCN, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            dilation = 2**l
            self.layers.append(
                DilatedResidualLayer(
                    num_filters,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        # Add MultiheadAttention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(num_filters)
        self.conv_out = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)

        # Apply self-attention
        # Convert from (batch, channels, seq_len) to (batch, seq_len, channels)
        out_trans = out.permute(0, 2, 1)
        attended_out, _ = self.attention(out_trans, out_trans, out_trans)
        attended_out = self.layer_norm(attended_out)
        # Convert back to (batch, channels, seq_len)
        out = attended_out.permute(0, 2, 1) + out  # Residual connection

        out = self.conv_out(out)
        return out


# Multi-Stage TCN model
class MSTCN(nn.Module):
    def __init__(
        self,
        num_stages,
        num_layers,
        num_classes,
        input_dim,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    ):
        super(MSTCN, self).__init__()
        self.stages = nn.ModuleList()
        # First stage with input dimensions
        self.stages.append(
            SSTCN(num_layers, input_dim, num_classes, num_filters, kernel_size, dropout)
        )
        # Refinement stages with number of classes as input
        for s in range(1, num_stages):
            self.stages.append(
                SSTCN(
                    num_layers,
                    num_classes,
                    num_classes,
                    num_filters,
                    kernel_size,
                    dropout,
                )
            )

    def forward(self, x):
        outputs = []
        out = x
        for stage in self.stages:
            out = stage(out)
            outputs.append(out)
        return outputs  # Return outputs from all stages


def MSTCN_Loss(outputs, targets, lambda_coef=0.15):
    total_loss = 0
    # Ensure targets are LongTensors
    targets = targets.long()

    # Cross-Entropy Loss
    ce_loss_fn = nn.CrossEntropyLoss()
    for output in outputs:
        batch_size, num_classes, seq_len = output.size()
        # Reshape output and targets for CrossEntropyLoss
        output = output.permute(0, 2, 1)  # [batch_size, seq_len, num_classes]
        output_reshaped = output.reshape(
            -1, num_classes
        )  # [batch_size*seq_len, num_classes]
        targets_reshaped = targets.reshape(-1)  # [batch_size*seq_len]

        # Compute classification loss
        ce_loss = ce_loss_fn(output_reshaped, targets_reshaped)

    # Smoothing Loss L_T-MSE with Weighted Time Differences
    log_probs = F.log_softmax(output, dim=2)  # [batch_size, seq_len, num_classes]
    log_probs = log_probs.permute(0, 2, 1)  # [batch_size, num_classes, seq_len]
    log_probs_t = log_probs[:, :, 1:]  # [batch_size, num_classes, seq_len-1]
    log_probs_t_minus_one = log_probs[:, :, :-1]  # [batch_size, num_classes, seq_len-1]

    # Calculate absolute difference
    delta = (log_probs_t - log_probs_t_minus_one).abs()

    # Generate weights for time steps
    seq_len = delta.size(2)  # Sequence length (seq_len-1 due to delta calculation)
    weights = torch.linspace(1.0, 0.1, steps=seq_len).to(
        delta.device
    )  # Linear decay from 1.0 to 0.1

    # Apply weights to the time differences
    weighted_delta = delta * weights.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and num_classes dimensions

    # Clamp values and compute weighted MSE loss
    weighted_delta = torch.clamp(weighted_delta, min=0, max=16)
    mse_loss = (weighted_delta**2).mean()

    # Compute total loss
    total_loss += ce_loss + lambda_coef * mse_loss

    return total_loss
