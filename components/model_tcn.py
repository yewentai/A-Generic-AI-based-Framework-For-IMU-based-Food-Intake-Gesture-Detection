import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        num_classes,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    ):
        super(TCN, self).__init__()
        self.conv_in = nn.Conv1d(input_dim, num_filters, kernel_size=1)
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
        return out  # [batch_size, num_classes, seq_len]


def TCN_Loss(outputs, targets):
    """
    outputs: [batch_size, num_classes, seq_len]
    targets: [batch_size, seq_len]
    """
    targets = targets.long()
    loss_ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_mse_fn = nn.MSELoss(reduction="none")
    batch_size, num_classes, seq_len = outputs.size()

    # Cross-entropy loss
    ce_loss = loss_ce_fn(
        outputs.transpose(2, 1).contiguous().view(-1, num_classes), targets.view(-1)
    )

    # Temporal smoothing loss
    if seq_len > 1:
        log_probs = F.log_softmax(outputs, dim=1)
        log_probs_next = log_probs[:, :, 1:]
        log_probs_prev = log_probs.detach()[:, :, :-1]  # stop gradient
        mse_loss = loss_mse_fn(log_probs_next, log_probs_prev)
        mse_loss = torch.clamp(mse_loss, min=0, max=16)
        mse_loss_mean = torch.mean(mse_loss)
    else:
        mse_loss_mean = torch.tensor(0.0, device=outputs.device)

    return ce_loss, mse_loss_mean
