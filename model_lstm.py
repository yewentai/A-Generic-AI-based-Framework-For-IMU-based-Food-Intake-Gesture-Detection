import torch
import torch.nn as nn

# Define a simple LSTM network
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True
        )

        # Forward through LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack and select only the outputs of the last time step for each sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(out[torch.arange(out.size(0)), lengths - 1])

        return out
    
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size

        # 定义权重矩阵
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)  # 初始化隐藏状态
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)  # 初始化记忆单元

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((h_t, x_t), dim=1)  # 拼接隐藏状态和输入

            f_t = torch.sigmoid(self.W_f(combined))  # 遗忘门
            i_t = torch.sigmoid(self.W_i(combined))  # 输入门
            c_hat_t = torch.tanh(self.W_c(combined))  # 候选记忆
            c_t = f_t * c_t + i_t * c_hat_t  # 更新记忆单元

            o_t = torch.sigmoid(self.W_o(combined))  # 输出门
            h_t = o_t * torch.tanh(c_t)  # 更新隐藏状态

        out = self.W_ho(h_t)  # 最后时间步的隐藏状态用于输出
        return out