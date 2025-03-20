import torch
import torch.nn as nn
import torch.nn.functional as F


# v1
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
    ):
        super(SSTCN, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                DilatedResidualLayer(
                    num_filters,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        self.conv_out = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)

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


def MSTCN_Loss(outputs, targets):
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

    return ce_loss, mse_loss


# v2
# Modified DilatedResidualLayer: ReLU -> 1x1 Conv -> dropout -> skip connection
class DilatedResidualLayer(nn.Module):
    def __init__(self, num_filters, dilation, kernel_size=3, dropout=0.3):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            num_filters, num_filters, kernel_size, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# Single-stage TCN module SSTCN
class SSTCN(nn.Module):
    def __init__(
        self,
        num_layers,
        in_channels,
        num_classes,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    ):
        super(SSTCN, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    num_filters, dilation=2**i, kernel_size=kernel_size, dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


# Multi-stage TCN model MSTCN
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
        # First stage takes the original features as input
        self.stages.append(
            SSTCN(num_layers, input_dim, num_classes, num_filters, kernel_size, dropout)
        )
        # Subsequent stages take the predictions (number of classes) from the previous stage as input
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
        # First stage output
        out = self.stages[0](x)
        outputs.append(out.unsqueeze(0))
        # Subsequent stages: apply softmax to the output of the previous stage before using it as input
        for stage in self.stages[1:]:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out.unsqueeze(0))
        # Concatenate outputs from all stages and adjust dimensions to (batch, stages, num_classes, time)
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2, 3)
        return outputs


def MSTCN_Loss(outputs, targets):
    # outputs shape: [batch_size, num_stages, num_classes, seq_len]
    # targets shape: [batch_size, seq_len]

    # Ensure targets are LongTensor
    targets = targets.long()

    loss_ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_mse_fn = nn.MSELoss(reduction="none")

    batch_size, num_stages, num_classes, seq_len = outputs.size()

    # Initialize losses
    ce_loss = 0
    mse_loss_mean = 0

    # Calculate loss for each stage
    for s in range(num_stages):
        stage_predictions = outputs[:, s, :, :]  # [batch_size, num_classes, seq_len]

        # Cross-entropy loss
        ce_loss += loss_ce_fn(
            stage_predictions.transpose(2, 1).contiguous().view(-1, num_classes),
            targets.view(-1),
        )

        # Temporal smoothness loss
        if seq_len > 1:
            mse_loss_value = loss_mse_fn(
                F.log_softmax(stage_predictions[:, :, 1:], dim=1),
                F.log_softmax(stage_predictions.detach()[:, :, :-1], dim=1),
            )
            mse_loss_value = torch.clamp(mse_loss_value, min=0, max=16)
            mse_loss_mean += torch.mean(mse_loss_value)

    # Average losses across stages
    ce_loss /= num_stages
    mse_loss_mean /= num_stages

    return ce_loss, mse_loss_mean
