import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, num_filters):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.key = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.value = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.scale = torch.sqrt(torch.FloatTensor([num_filters])).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = torch.matmul(Q.permute(0, 2, 1), K) / self.scale
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, V.permute(0, 2, 1))
        return out.permute(0, 2, 1)


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

        # Add the self-attention layer
        self.attention = SelfAttention(num_filters)

        self.conv_out = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)

        # Apply self-attention
        # out = self.attention(out) + out  # Residual connection

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

        # Smoothing Loss L_T-MSE
        log_probs = F.log_softmax(output, dim=2)  # [batch_size, seq_len, num_classes]
        log_probs = log_probs.permute(0, 2, 1)  # [batch_size, num_classes, seq_len]
        log_probs_t = log_probs[:, :, 1:]  # [batch_size, num_classes, seq_len-1]
        log_probs_t_minus_one = log_probs[
            :, :, :-1
        ]  # [batch_size, num_classes, seq_len-1]
        delta = (log_probs_t - log_probs_t_minus_one).abs()
        delta = torch.clamp(delta, min=0, max=16)
        mse_loss = (delta**2).mean()

        total_loss += ce_loss + lambda_coef * mse_loss

    return total_loss


# for p in predictions:
#     ce_loss = loss_ce(p.transpose(2, 1).contiguous().view(-1, num_class), oneBatchLabel2.view(-1, num_class))
#     mse_loss = 0.15 * torch.mean(torch.clamp(loss_mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
