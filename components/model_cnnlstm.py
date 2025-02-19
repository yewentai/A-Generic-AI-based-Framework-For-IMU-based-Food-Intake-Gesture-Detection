import torch
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # Removed pooling layers
        x = F.relu(self.conv3(x))
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.sigmoid(self.fc(x))
        return x


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv_net = ConvNet()
        self.recurrent_net = LSTM()

    def forward(self, x):
        x = self.conv_net(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        x = self.recurrent_net(x)
        return x
