import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import pickle
from model_vae import VAE, loss_function

def train_vae(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0

    for data in train_loader:
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Average Training Loss: {avg_loss:.4f}")

def generate_samples(model, num_samples, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
    return samples.cpu().numpy()

def main():
    # Configuration
    input_dim = 6 * 128  # Example for time-series with 6 features and sequence length of 128
    hidden_dim = 512
    latent_dim = 20
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-3
    num_samples = 100  # Number of samples to generate

    # Load existing data for training the VAE
    X_path = "./dataset/pkl_data/DX_I_X.pkl"
    with open(X_path, "rb") as f:
        X = pickle.load(f)

    # Preprocess and normalize data
    X = [x.flatten() for x in X]
    X = np.array(X)
    X = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]

    # Create Dataset and DataLoader
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train VAE
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_vae(model, train_loader, optimizer, device)

    # Generate new augmented samples
    new_samples = generate_samples(model, num_samples, device)
    print("Generated Samples:", new_samples)

if __name__ == "__main__":
    main()