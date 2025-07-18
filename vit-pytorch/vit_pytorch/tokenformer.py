import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    """Helper function to ensure t is a tuple."""
    return t if isinstance(t, tuple) else (t, t)

# classes

class PattentionLayer(nn.Module):
    """
    The core Token-Parameter Attention (Pattention) layer from the Tokenformer paper.
    This layer replaces a standard nn.Linear projection. Instead of a fixed weight matrix,
    it uses a set of learnable key-value parameter tokens.
    """
    def __init__(self, dim_in, dim_out, num_param_tokens, device='cpu'):
        """
        Initializes the Pattention layer.
        Args:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_param_tokens (int): The number of learnable parameter tokens (n in the paper).
            device (str): The device to place tensors on.
        """
        super().__init__()
        self.scale = dim_in ** -0.5
        self.device = device

        # Learnable parameter tokens for keys and values
        self.key_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_in))
        self.value_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_out))

    def forward(self, x):
        """
        Forward pass for Pattention.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_tokens, dim_in).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_input_tokens, dim_out).
        """
        # Calculate similarity between input tokens (queries) and key parameter tokens
        similarity = torch.matmul(x, self.key_param_tokens.T) * self.scale

        # Apply Theta (Modified Softmax): L2 norm + GeLU activation
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)

        # Compute the weighted sum of value parameter tokens
        out = torch.matmul(attn_weights, self.value_param_tokens)
        return out

    def grow(self, num_new_tokens):
        """
        Expands the layer by adding new, randomly initialized parameter tokens.
        Args:
            num_new_tokens (int): The number of new key-value pairs to add.
        """
        # Get dimensions and device from existing parameters
        dim_in = self.key_param_tokens.shape[1]
        dim_out = self.value_param_tokens.shape[1]
        
        # Create new parameter tokens
        new_key_tokens = torch.randn(num_new_tokens, dim_in, device=self.device)
        new_value_tokens = torch.randn(num_new_tokens, dim_out, device=self.device)
        
        # Concatenate new tokens to the existing ones
        self.key_param_tokens = nn.Parameter(torch.cat([self.key_param_tokens.data, new_key_tokens], dim=0))
        self.value_param_tokens = nn.Parameter(torch.cat([self.value_param_tokens.data, new_value_tokens], dim=0))
        print(f"Grew Pattention layer. New total parameter tokens: {self.key_param_tokens.shape[0]}")


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
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.initial_ffn_param_tokens = mlp_dim 
        self.initial_attn_param_tokens = dim 

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            PattentionLayer(patch_dim, dim, num_param_tokens=dim, device=device),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TokenformerEncoder(dim, depth, heads, dim_head, mlp_dim, self.initial_ffn_param_tokens, self.initial_attn_param_tokens, dropout, device=device)
        self.pool = pool
        self.to_latent = nn.Identity()

        # Multi-head classifier for continual learning
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
        
        # Use the correct head for the given task
        return self.mlp_heads[task_id](x)

    def grow(self, ffn_growth_factor=2, attn_growth_factor=2):
        """ Grows the entire model by expanding all Pattention layers. """
        print("\n--- Growing Model ---")
        ffn_new_tokens = self.initial_ffn_param_tokens // ffn_growth_factor
        attn_new_tokens = self.initial_attn_param_tokens // attn_growth_factor

        for module in self.modules():
            if isinstance(module, PattentionLayer):
                # Determine if it's an attention or FFN Pattention layer to apply the correct growth factor
                if module.key_param_tokens.shape[1] == self.initial_ffn_param_tokens:
                     module.grow(ffn_new_tokens)
                elif module.key_param_tokens.shape[1] == self.initial_attn_param_tokens:
                     module.grow(attn_new_tokens)
                else: # For embedding and mlp_head layers
                     module.grow(attn_new_tokens)
        print("--- Model Growth Complete ---\n")