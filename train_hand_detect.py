import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
from utils import IMUDataset

# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 训练自编码器
def train_autoencoder(data, epochs=100, batch_size=32):
    input_dim = data.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    return model

# 检测异常
def detect_anomalies(model, data, threshold):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        mse = nn.MSELoss(reduction='none')
        reconstruction_errors = mse(outputs, data).mean(axis=1)
    
    anomalies = reconstruction_errors > threshold
    return anomalies

# 修正数据
def correct_data(dataset, anomalies):
    correction_matrix = np.array([
        [-1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, -1]
    ])
    
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            x, y = dataset[i]
            x = x.numpy()
            x = np.dot(x, correction_matrix)
            dataset.data[i] = x

    return dataset

# 主函数
def main(X, Y):
    # 创建数据集
    dataset = IMUDataset(X, Y)

    # 转换为PyTorch张量
    data = torch.FloatTensor(dataset)

    # 训练自编码器
    model = train_autoencoder(data)

    # 检测异常
    anomalies = detect_anomalies(model, data, threshold=0.1)

    # 修正数据
    corrected_dataset = correct_data(dataset, anomalies)

    return corrected_dataset, anomalies

# 使用示例
if __name__ == "__main__":
    # 加载数据
    X_path = "./dataset/pkl_data/DX_I_X.pkl"
    Y_path = "./dataset/pkl_data/DX_I_Y.pkl"
    with open(X_path, "rb") as f:
        X = pickle.load(f)
    with open(Y_path, "rb") as f:
        Y = pickle.load(f)

    corrected_dataset, detected_anomalies = main(X, Y)

    print(f"检测到的异常 session 数量: {sum(detected_anomalies)}")
    print(f"总 session 数量: {len(detected_anomalies)}")