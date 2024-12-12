import torch
import torch.nn as nn

# Define a simple GRU network
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True
        )

        # Forward through GRU
        packed_output, _ = self.gru(packed_input)

        # Unpack and select only the outputs of the last time step for each sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(out[torch.arange(out.size(0)), lengths - 1])

        return out
    
class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomGRU, self).__init__()
        self.hidden_size = hidden_size

        # 定义权重矩阵
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # 更新门
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)  # 重置门
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)  # 候选隐藏状态
        self.W_ho = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)  # 初始化隐藏状态

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((h_t, x_t), dim=1)  # 拼接隐藏状态和输入

            z_t = torch.sigmoid(self.W_z(combined))  # 更新门
            r_t = torch.sigmoid(self.W_r(combined))  # 重置门
            h_hat_t = torch.tanh(
                self.W_h(torch.cat((r_t * h_t, x_t), dim=1))
            )  # 候选隐藏状态
            h_t = (1 - z_t) * h_t + z_t * h_hat_t  # 更新隐藏状态

        out = self.W_ho(h_t)  # 最后时间步的隐藏状态用于输出
        return out