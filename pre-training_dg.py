# DG_layer.py
import torch
from torch import nn
# pretrain_dg.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100 # Pre-training on MNIST is fine for this
from torchvision import transforms
from tqdm import tqdm
import os

class SparseImageAutoencoder(nn.Module):
    """
    An autoencoder that takes an image, maps it to a large, sparse latent space
    using a k-Winners-Take-All mechanism, and reconstructs the original image.
    """
    def __init__(self, image_size=28, channels=1, latent_dim=1024, k=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.k = int(k)
        
        # Encoder: Converts the image to a latent vector
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1), # -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        
        # Decoder: Reconstructs the image from the latent vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1), # -> 28x28
            nn.Sigmoid() # Use Sigmoid to ensure output pixels are between 0 and 1
        )

    def k_winner_take_all(self, x):
        """Applies k-Winners-Take-All activation to the latent vector."""
        if self.k >= self.latent_dim:
            return x
        
        top_k_values, top_k_indices = torch.topk(x, self.k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, top_k_indices, 1)
        return x * mask

    def encode(self, img):
        """
        Encodes an image to its sparse latent representation.
        This is the method that will be used by the ContinualLearner.
        """
        latent_vector = self.encoder(img)
        sparse_latent_vector = self.k_winner_take_all(latent_vector)
        return sparse_latent_vector

    def forward(self, img):
        """
        Performs a full autoencoder pass: encode -> decode.
        This is used during the pre-training phase.
        """
        sparse_latent_vector = self.encode(img)
        reconstructed_img = self.decoder(sparse_latent_vector)
        return reconstructed_img




def pretrain():
    config = {
        "epochs": 10, "batch_size": 256, "lr": 1e-3,
        "image_size": 28, 
        "latent_dim": 1024, "dg_k": 50, # k = 5% of 1024 is ~50
        "save_path": "dg_autoencoder_pretrained.pth"
    }
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # No normalization needed if using Sigmoid output and MSE loss
    ])
    # Using MNIST for pre-training is fine, as it's an image dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    model = SparseImageAutoencoder(
        image_size=config["image_size"],
        latent_dim=config["latent_dim"],
        k=config["dg_k"],
        channels=1
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss() # Mean Squared Error for reconstruction

    print("🚀 Starting Sparse Autoencoder Pre-training...")
    for epoch in range(config["epochs"]):
        loop = tqdm(train_loader, leave=True, desc=f"Pre-train Epoch {epoch+1}")
        total_loss = 0
        for images, _ in loop:
            images = images.to(DEVICE)
            
            optimizer.zero_grad()
            reconstructed_images = model(images)
            loss = criterion(reconstructed_images, images) # Compare reconstructed with original
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Avg Reconstruction Loss: {total_loss / len(train_loader):.6f}")

    print(f"✅ Pre-training finished. Saving weights to {config['save_path']}")
    torch.save(model.state_dict(), config['save_path'])

if __name__ == '__main__':
    pretrain()