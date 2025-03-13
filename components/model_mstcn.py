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
    # 保证 targets 为 LongTensor
    targets = targets.long()
    loss_ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_mse_fn = nn.MSELoss(reduction="none")

    total_loss = 0
    # 遍历所有 stage 的预测，计算总 loss
    for p in outputs:
        # p 的尺寸为 [batch_size, num_classes, seq_len]
        # 交叉熵损失：先将 p 转置为 [batch_size, seq_len, num_classes]，
        # 再 reshape 为 [batch_size*seq_len, num_classes]；targets reshape 为 [batch_size*seq_len]
        ce_loss = loss_ce_fn(
            p.transpose(2, 1).contiguous().view(-1, p.size(1)), targets.view(-1)
        )

        # 平滑损失：对时间维度上相邻时刻的 log_softmax 结果计算 MSE
        mse_loss_value = loss_mse_fn(
            F.log_softmax(p[:, :, 1:], dim=1),
            F.log_softmax(p.detach()[:, :, :-1], dim=1),
        )
        # 对 mse_loss 进行 clamp 操作，然后求均值
        mse_loss_value = torch.clamp(mse_loss_value, min=0, max=16)
        mse_loss_mean = torch.mean(mse_loss_value)

    return ce_loss, mse_loss_mean
