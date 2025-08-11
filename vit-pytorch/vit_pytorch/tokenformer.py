# tokenformer.py

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# PattentionLayer, TokenformerAttention, TokenformerFeedForward, TokenformerEncoder
# remain EXACTLY THE SAME as your previous version. They are the core building blocks.

# ... (Paste PattentionLayer, TokenformerFeedForward, TokenformerAttention, TokenformerEncoder here) ...
class PattentionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_param_tokens, device='cpu'):
        super().__init__()
        self.scale = dim_in ** -0.5
        self.device = device
        self.key_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_in))
        self.value_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_out))
        self.register_buffer('key_grad_mask', torch.ones_like(self.key_param_tokens))
        self.register_buffer('value_grad_mask', torch.ones_like(self.value_param_tokens))
        self.growth_indices = []

    def forward(self, x):
        similarity = torch.matmul(x, self.key_param_tokens.T) * self.scale
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        self.attn_weights = attn_weights
        out = torch.matmul(attn_weights, self.value_param_tokens)
        return out

    def grow(self, num_new_tokens):
        dim_in = self.key_param_tokens.shape[1]
        dim_out = self.value_param_tokens.shape[1]
        num_old_tokens = self.key_param_tokens.shape[0]
        self.growth_indices.append(num_old_tokens)
        self.key_grad_mask.fill_(0)
        self.value_grad_mask.fill_(0)
        new_key_tokens = torch.randn(num_new_tokens, dim_in, device=self.device) * 0.01
        new_value_tokens = torch.randn(num_new_tokens, dim_out, device=self.device) * 0.01
        new_key_mask = torch.ones_like(new_key_tokens)
        new_value_mask = torch.ones_like(new_value_tokens)
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

### --- NEW ARCHITECTURE --- ###

class GrowingTokenformer(nn.Module):
    """
    This module contains the part of the network that grows: the Tokenformer encoder
    and the multi-head output layer. It operates on feature vectors, not images.
    """
    def __init__(self, *, dim, depth, heads, mlp_dim, num_tasks, classes_per_task, device='cpu'):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        
        # The Tokenformer encoder that will process feature vectors
        self.transformer = TokenformerEncoder(dim, depth, heads, dim, mlp_dim, mlp_dim, dim, device=device)
        
        # The multi-head output layer that is selected by task_id
        self.mlp_heads = nn.ModuleList([
            PattentionLayer(dim, classes_per_task, num_param_tokens=classes_per_task * 4, device=device) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id):
        # The encoder expects a sequence. We unsqueeze to add a dummy sequence dimension.
        x = x.unsqueeze(1)
        x = self.transformer(x)
        # Squeeze back to a feature vector
        x = x.squeeze(1)
        return self.mlp_heads[task_id](x)

    def grow(self):
        """ Grows the transformer by expanding its Pattention layers. """
        # NOTE: Growth factors are now hardcoded for simplicity, could be passed in.
        attn_new_tokens = self.dim // 2

        # We only grow the transformer, not the output heads
        for module in self.transformer.modules():
            if isinstance(module, PattentionLayer):
                module.grow(attn_new_tokens)

class ContinualLearner(nn.Module):
    """
    The main model that combines a frozen backbone with the growing Tokenformer module.
    """
    def __init__(self, *, dim, depth, heads, mlp_dim, num_tasks, classes_per_task, device='cpu'):
        super().__init__()
        
        # 1. Load and freeze the pretrained backbone (ResNet-18)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone_out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove the final classification layer

        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Adapter layer to match backbone output to tokenformer input dimension
        self.adapter = nn.Linear(backbone_out_features, dim)
        
        # 3. The growing part of our model
        self.growing_module = GrowingTokenformer(
            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
            num_tasks=num_tasks, classes_per_task=classes_per_task, device=device
        )
        
    def forward(self, img, task_id):
        self.backbone.eval() # Ensure backbone is always in eval mode (for batchnorm)
        features = self.backbone(img)
        adapted_features = self.adapter(features)
        return self.growing_module(adapted_features, task_id)

    def grow(self):
        print("\n--- Growing Model (Tokenformer Module) ---")
        self.growing_module.grow()
        print("--- Model Growth Complete ---")