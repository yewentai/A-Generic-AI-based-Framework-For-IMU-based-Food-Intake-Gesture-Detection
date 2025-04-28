import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 定义VAE模型类
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # 编码器
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.encoder_fc1(x))
        mean = self.encoder_fc2_mean(h1)
        logvar = self.encoder_fc2_logvar(h1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        return torch.sigmoid(self.decoder_fc2(h1))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar


# 定义计算KL散度的函数
def kl_divergence(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


# 定义计算ELBO的函数，使用MSE作为重建损失函数
def elbo(recon_x, x, mean, logvar, beta=1):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = kl_divergence(mean, logvar)
    return reconstruction_loss, kl_loss


# 生成示例数据
input_dim = 10
num_samples = 100
data = torch.randn(num_samples, input_dim)
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=10)

# 初始化VAE模型并定义优化器
hidden_dim = 20
latent_dim = 5
vae_model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae_model.parameters(), lr=0.001)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    for batch_data in dataloader:
        x = batch_data[0]

        optimizer.zero_grad()

        recon_x, mean, logvar = vae_model(x)

        # 计算ELBO
        recon_loss, kl_loss = elbo(recon_x, x, mean, logvar)

        beta = min(1.0, epoch / 100)  # Gradually increase beta up to 1.0 over 50 epochs
        loss = recon_loss - beta * kl_loss

        # 计算KL散度
        kl_value = kl_divergence(mean, logvar)

        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}")
