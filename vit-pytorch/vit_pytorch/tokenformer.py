# model.py
import torch
from torch import nn
import torch.nn.functional as F
# from DG_layer import SparseImageAutoencoder # <-- Import the new autoencoder

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


class PattentionLayer(nn.Module):
    # (This class is unchanged)
    def __init__(self, dim_in, dim_out, num_initial_tokens, device='cpu'):
        super().__init__()
        self.device = device
        self.key_param_tokens = nn.Parameter(torch.randn(num_initial_tokens, dim_in))
        self.value_param_tokens = nn.Parameter(torch.randn(num_initial_tokens, dim_out))
        self.register_buffer('grad_mask', torch.ones(num_initial_tokens, 1, device=device))
        self.growth_indices = [0]

    def forward(self, x):
        num_total_tokens = self.key_param_tokens.shape[0]
        if num_total_tokens == 0:
            return torch.zeros(x.shape[:-1] + (self.value_param_tokens.shape[1],), device=self.device)
        similarity = torch.matmul(x, self.key_param_tokens.T)
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        return torch.matmul(attn_weights, self.value_param_tokens)

    def grow(self, num_new_tokens):
        device = self.key_param_tokens.device
        dim_in, dim_out = self.key_param_tokens.shape[1], self.value_param_tokens.shape[1]
        old_keys, old_values = self.key_param_tokens.data, self.value_param_tokens.data
        random_indices = torch.randint(0, old_keys.shape[0], (num_new_tokens,), device=device)
        base_new_keys = old_keys[random_indices] + torch.randn(num_new_tokens, dim_in, device=device) * 0.01
        base_new_values = old_values[random_indices] + torch.randn(num_new_tokens, dim_out, device=device) * 0.01
        self.grad_mask.fill_(0)
        new_mask = torch.ones(num_new_tokens, 1, device=device)
        self.key_param_tokens = nn.Parameter(torch.cat([old_keys, base_new_keys], dim=0))
        self.value_param_tokens = nn.Parameter(torch.cat([old_values, base_new_values], dim=0))
        self.grad_mask = torch.cat([self.grad_mask.data, new_mask], dim=0)
        self.growth_indices.append(old_keys.shape[0])


class TokenformerEncoder(nn.Module):
    # (This class is unchanged)
    def __init__(self, dim, depth, num_initial_tokens, device):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PattentionLayer(dim, dim, num_initial_tokens, device),
                nn.LayerNorm(dim),
            ]))
    def forward(self, x):
        for attn, norm in self.layers:
            x = attn(x) + x
            x = norm(x)
        return x

class ContinualLearner(nn.Module):
    def __init__(self, *, image_size, depth,
                 num_initial_tokens, latent_dim, dg_k, num_total_classes,
                 channels=1, device='cpu'):
        super().__init__()
        
        # --- MODIFIED: The feature extractor is now the pre-trained autoencoder ---
        self.feature_extractor = SparseImageAutoencoder(
            image_size=image_size,
            channels=channels,
            latent_dim=latent_dim,
            k=dg_k
        )
        
        # The transformer now takes the latent vector as input
        self.transformer = TokenformerEncoder(latent_dim, depth, num_initial_tokens, device)
        self.output_head = nn.Linear(latent_dim, num_total_classes)

    def forward(self, img):
        # 1. Get the sparse latent vector from the frozen autoencoder's encoder
        with torch.no_grad(): # Ensure feature extractor is not trained
            sparse_latent_vector = self.feature_extractor.encode(img)
        
        # 2. Add a sequence dimension for the transformer
        x = sparse_latent_vector.unsqueeze(1)
        
        # 3. Process through the transformer
        x = self.transformer(x)
        
        # 4. Get final features and classify
        features = x.squeeze(1)
        return self.output_head(features)

    def grow_transformer(self, num_new_tokens):
        for module in self.transformer.modules():
            if isinstance(module, PattentionLayer):
                module.grow(num_new_tokens)