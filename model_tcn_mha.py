import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Chops off extra padding added during convolution to ensure causality."""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single temporal block with two layers of dilated causal convolutions."""

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
        self.chomp1 = Chomp1d(padding)
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
        self.chomp2 = Chomp1d(padding)
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
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
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
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
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
    ):
        super().__init__()

        # TCN Module
        self.tcn = TemporalConvNet(input_dim, hidden_dim, kernel_size, num_layers)

        # Linear projection to d_model dimension for MHA
        self.projection = nn.Linear(hidden_dim, d_model)

        # MHA Module
        self.mha = MultiHeadAttention(d_model, num_heads)

        # FCN Module
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # TCN Module
        x = self.tcn(x)

        # Project to d_model dimension
        x = self.projection(x)

        # MHA Module
        x = self.mha(x)

        # FCN Module
        x = self.fcn(x)

        return x


def TCNMHA_Loss(output, target, lambda_coef=0.15):
    total_loss = 0
    # Ensure target are LongTensors
    target = target.long()

    # Cross-Entropy Loss
    ce_loss_fn = nn.CrossEntropyLoss()
    batch_size, num_classes, seq_len = output.size()
    # Reshape output and target for CrossEntropyLoss
    output = output.permute(0, 2, 1)  # [batch_size, seq_len, num_classes]
    output_reshaped = output.reshape(
        -1, num_classes
    )  # [batch_size*seq_len, num_classes]
    target_reshaped = target.reshape(-1)  # [batch_size*seq_len]

    # Compute classification loss
    ce_loss = ce_loss_fn(output_reshaped, target_reshaped)

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
