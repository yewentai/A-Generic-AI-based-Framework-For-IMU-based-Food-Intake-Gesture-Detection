import torch
import torch.nn as nn
import torch.nn.functional as F


# 修改后的 DilatedResidualLayer：ReLU -> 1x1卷积 -> dropout -> skip connection
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


# 单阶段 TCN 模块 SSTCN
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


# 多阶段 TCN 模型 MSTCN
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
        # 第一个阶段输入原始特征
        self.stages.append(
            SSTCN(num_layers, input_dim, num_classes, num_filters, kernel_size, dropout)
        )
        # 后续阶段输入为上个阶段的预测（类别数）
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
        # 第一阶段直接输出
        out = self.stages[0](x)
        outputs.append(out.unsqueeze(0))
        # 后续阶段：先对上一阶段输出做 softmax 再作为输入
        for stage in self.stages[1:]:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out.unsqueeze(0))
        # 拼接各阶段输出，并调整维度为 (batch, stages, num_classes, time)
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
