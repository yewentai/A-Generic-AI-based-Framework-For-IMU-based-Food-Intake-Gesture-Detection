import math
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """A single temporal block with two layers of dilated non-causal convolutions."""

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        # First convolutional layer
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Skip connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs
            else nn.Identity()
        )
        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )
        self.relu = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if isinstance(self.downsample, nn.Conv1d):
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """A Temporal Convolutional Network (TCN) with multiple temporal blocks."""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Adjust padding for "same" padding
            padding = (kernel_size - 1) * dilation_size // 2

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Add positional encoding
        x = self.pos_encoder(x)

        # MultiHead Attention expects (seq_len, batch_size, dim)
        x = x.transpose(0, 1)

        # Self-attention
        attn_output, _ = self.multihead_attn(x, x, x)

        # Add & Norm
        x = self.norm(x + self.dropout(attn_output))

        # Return to (batch_size, seq_len, dim)
        return x.transpose(0, 1)


class TCNMHA(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_classes=3,
        num_heads=8,
        d_model=128,
        kernel_size=3,
        num_layers=9,
        dropout=0.2,
    ):
        super().__init__()

        # TCN Module
        self.tcn = TemporalConvNet(
            input_dim, [hidden_dim] * num_layers, kernel_size, dropout
        )

        # Linear projection to d_model dimension for MHA
        self.projection = nn.Linear(hidden_dim, d_model)

        # MHA Module
        self.mha = MultiHeadAttention(d_model, num_heads)

        # FCN Module
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # TCN Module
        x = self.tcn(x)  # Output shape: [batch_size, hidden_dim, seq_len]

        # Transpose to match Linear layer input
        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]

        # Project to d_model dimension
        x = self.projection(x)  # [batch_size, seq_len, d_model]

        # MHA Module
        x = self.mha(x)  # [batch_size, seq_len, d_model]

        # FCN Module
        x = self.fcn(x)  # [batch_size, seq_len, num_classes]

        return x


def TCNMHA_Loss(output, target, lambda_coef):
    # Ensure target is a LongTensor
    target = target.long()

    # Reshape output for CrossEntropyLoss
    batch_size, seq_len, num_classes = output.size()
    output_reshaped = output.view(-1, num_classes)  # [batch_size*seq_len, num_classes]
    target_reshaped = target.view(-1)  # [batch_size*seq_len]

    # Cross-Entropy Loss
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(output_reshaped, target_reshaped)

    # Smoothing Loss L_T-MSE with Weighted Time Differences
    log_probs = F.log_softmax(output, dim=2)  # [batch_size, seq_len, num_classes]
    log_probs_t = log_probs[:, 1:, :]  # [batch_size, seq_len-1, num_classes]
    log_probs_t_minus_one = log_probs[:, :-1, :]  # [batch_size, seq_len-1, num_classes]

    # Calculate absolute difference
    delta = (log_probs_t - log_probs_t_minus_one).abs()

    # Generate weights for time steps
    seq_len = delta.size(1)  # Sequence length (seq_len-1 due to delta calculation)
    weights = torch.linspace(1.0, 0.1, steps=seq_len).to(delta.device)

    # Apply weights to the time differences
    weighted_delta = delta * weights.unsqueeze(0).unsqueeze(-1)

    # Clamp values and compute weighted MSE loss
    mse_loss = (weighted_delta**2).mean()

    # Compute total loss
    total_loss = ce_loss + lambda_coef * mse_loss

    return total_loss
