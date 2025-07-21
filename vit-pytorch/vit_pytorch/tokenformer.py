import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm
import numpy as np

# helpers

def pair(t):
    """Helper function to ensure t is a tuple."""
    return t if isinstance(t, tuple) else (t, t)

def count_parameters(model):
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def apply_grad_mask_hook(grad, mask):
    """A gradient hook that applies a mask to the gradients."""
    return grad * mask

# classes

class PattentionLayer(nn.Module):
    """
    The core Token-Parameter Attention (Pattention) layer, modified to support growth
    with gradient masks.
    """
    def __init__(self, dim_in, dim_out, num_param_tokens, device='cpu'):
        super().__init__()
        self.scale = dim_in ** -0.5
        self.device = device

        # Initialize parameters as a single tensor
        self.key_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_in))
        self.value_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_out))

        # Initialize gradient masks as buffers
        self.register_buffer('key_grad_mask', torch.ones_like(self.key_param_tokens))
        self.register_buffer('value_grad_mask', torch.ones_like(self.value_param_tokens))


    def forward(self, x):
        """
        Forward pass for Pattention.
        """
        similarity = torch.matmul(x, self.key_param_tokens.T) * self.scale
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        out = torch.matmul(attn_weights, self.value_param_tokens)
        return out

    def grow(self, num_new_tokens):
        """
        Expands the layer by adding new parameter tokens and updating the gradient mask.
        """
        # Get dimensions and device from existing parameters
        dim_in = self.key_param_tokens.shape[1]
        dim_out = self.value_param_tokens.shape[1]
        
        # --- Freeze existing parameters by setting their mask to 0 ---
        self.key_grad_mask.fill_(0)
        self.value_grad_mask.fill_(0)

        # --- Create new parameter tokens ---
        new_key_tokens = torch.randn(num_new_tokens, dim_in, device=self.device)
        new_value_tokens = torch.randn(num_new_tokens, dim_out, device=self.device)
        
        # --- Create masks for the new tokens ---
        new_key_mask = torch.ones_like(new_key_tokens)
        new_value_mask = torch.ones_like(new_value_tokens)

        # --- Concatenate old and new parameters/masks ---
        self.key_param_tokens = nn.Parameter(torch.cat([self.key_param_tokens.data, new_key_tokens], dim=0))
        self.value_param_tokens = nn.Parameter(torch.cat([self.value_param_tokens.data, new_value_tokens], dim=0))
        
        self.key_grad_mask = torch.cat([self.key_grad_mask, new_key_mask], dim=0)
        self.value_grad_mask = torch.cat([self.value_grad_mask, new_value_mask], dim=0)


class TokenformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_num_param_tokens, dropout = 0., device='cpu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            PattentionLayer(dim, hidden_dim, num_param_tokens=ffn_num_param_tokens, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
            PattentionLayer(hidden_dim, dim, num_param_tokens=dim, device=device),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TokenformerAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, attn_num_param_tokens=None, dropout = 0., device='cpu'):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        if attn_num_param_tokens is None:
            attn_num_param_tokens = inner_dim
            
        self.to_q = PattentionLayer(dim, inner_dim, num_param_tokens=attn_num_param_tokens, device=device)
        self.to_k = PattentionLayer(dim, inner_dim, num_param_tokens=attn_num_param_tokens, device=device)
        self.to_v = PattentionLayer(dim, inner_dim, num_param_tokens=attn_num_param_tokens, device=device)
        self.to_out = PattentionLayer(inner_dim, dim, num_param_tokens=dim, device=device) if project_out else nn.Identity()

    def forward(self, x):
        x_norm = self.norm(x)
        q, k, v = self.to_q(x_norm), self.to_k(x_norm), self.to_v(x_norm)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TokenformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, ffn_num_param_tokens, attn_num_param_tokens, dropout = 0., device='cpu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TokenformerAttention(dim, heads=heads, dim_head=dim_head, attn_num_param_tokens=attn_num_param_tokens, dropout=dropout, device=device),
                TokenformerFeedForward(dim, mlp_dim, ffn_num_param_tokens=ffn_num_param_tokens, dropout=dropout, device=device)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TokenformerViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_tasks, classes_per_task, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., device='cpu'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.mlp_dim = mlp_dim
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            PattentionLayer(patch_dim, dim, num_param_tokens=dim, device=device),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TokenformerEncoder(dim, depth, heads, dim_head, mlp_dim, mlp_dim, dim, dropout, device=device)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_heads = nn.ModuleList([
            PattentionLayer(dim, classes_per_task, num_param_tokens=classes_per_task * 4, device=device) for _ in range(num_tasks)
        ])

    def forward(self, img, task_id):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_heads[task_id](x)

    def grow(self, ffn_growth_factor=2, attn_growth_factor=2):
        """ Grows the entire model by expanding all Pattention layers. """
        print("\n--- Growing Model ---")
        ffn_new_tokens = self.mlp_dim // ffn_growth_factor
        attn_new_tokens = self.dim // attn_growth_factor

        for module in self.modules():
            if isinstance(module, PattentionLayer):
                 module.grow(attn_new_tokens)
        print("--- Model Growth Complete ---")