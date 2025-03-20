import torch
import torch.nn as nn
import torch.nn.functional as F


# Modified DilatedResidualLayer
class DilatedResidualLayer(nn.Module):
    def __init__(self, num_filters, dilation, kernel_size=3, dropout=0.3):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            num_filters, num_filters, kernel_size, padding=dilation, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# Single-stage TCN model (SSTCN)
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


# Multi-stage TCN model (MSTCN)
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
        # The first stage takes the original features as input
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
        # Output of the first stage
        out = self.stages[0](x)
        outputs.append(out.unsqueeze(0))
        # Subsequent stages: apply softmax normalization to the output of the previous stage before input
        for stage in self.stages[1:]:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out.unsqueeze(0))
        outputs = torch.cat(
            outputs, dim=0
        )  # shape: [num_stages, batch, num_classes, seq_len]
        outputs = outputs.permute(
            1, 0, 2, 3
        )  # shape: [batch, num_stages, num_classes, seq_len]
        return outputs


# Loss function for MSTCN
def MSTCN_Loss(outputs, targets):
    """
    outputs shape: [batch_size, num_stages, num_classes, seq_len]
    targets shape: [batch_size, seq_len]
    """
    targets = targets.long()
    loss_ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_mse_fn = nn.MSELoss(reduction="none")
    batch_size, num_stages, num_classes, seq_len = outputs.size()
    ce_loss = 0
    mse_loss_mean = 0

    # Compute loss for each stage
    for s in range(num_stages):
        stage_predictions = outputs[:, s, :, :]  # [batch_size, num_classes, seq_len]
        # Cross-entropy loss
        ce_loss += loss_ce_fn(
            stage_predictions.transpose(2, 1).contiguous().view(-1, num_classes),
            targets.view(-1),
        )
        # Temporal smoothing loss (only computed when seq_len > 1)
        if seq_len > 1:
            mse_loss_value = loss_mse_fn(
                F.log_softmax(stage_predictions[:, :, 1:], dim=1),
                F.log_softmax(stage_predictions.detach()[:, :, :-1], dim=1),
            )
            mse_loss_value = torch.clamp(mse_loss_value, min=0, max=16)
            mse_loss_mean += torch.mean(mse_loss_value)
    # Average the loss across stages
    ce_loss /= num_stages
    mse_loss_mean /= num_stages

    return ce_loss, mse_loss_mean
