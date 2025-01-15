import torch
import torch.nn as nn
import math


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class TCNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=9):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, 1)

        # Stack dilated convolution layers
        self.dilated_convs = nn.ModuleList(
            [
                DilatedConvBlock(hidden_dim, hidden_dim, kernel_size, dilation=2**i)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        # Input shape: (batch_size, T, input_dim)
        # Convert to (batch_size, input_dim, T) for convolution
        x = x.transpose(1, 2)

        x = self.input_conv(x)

        for conv in self.dilated_convs:
            x = conv(x)

        # Convert back to (batch_size, T, hidden_dim)
        return x.transpose(1, 2)


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


class MHAModule(nn.Module):
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
        self.tcn = TCNModule(input_dim, hidden_dim, kernel_size, num_layers)

        # Linear projection to d_model dimension for MHA
        self.projection = nn.Linear(hidden_dim, d_model)

        # MHA Module
        self.mha = MHAModule(d_model, num_heads)

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


# Custom loss function combining classification and smoothing loss
class TCNMHALoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Classification loss (Cross Entropy)
        ce_loss = self.ce_loss(pred.view(-1, pred.size(-1)), target.view(-1))

        # Smoothing loss (Truncated MSE)
        # Calculate temporal differences
        temp_diff = torch.diff(pred, dim=1)
        smoothing_loss = torch.mean(torch.clamp(temp_diff**2, min=0, max=1))

        # Combine losses
        total_loss = ce_loss + self.alpha * smoothing_loss

        return total_loss


# Example usage:
def create_model(input_dim, **kwargs):
    model = TCNMHA(input_dim, **kwargs)
    criterion = TCNMHALoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    return model, criterion, optimizer
