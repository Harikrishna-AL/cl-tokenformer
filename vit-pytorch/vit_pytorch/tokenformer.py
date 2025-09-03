# tokenformer.py

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange, repeat

# --- MODIFIED: PattentionLayer with Agnostic/Specific parameter separation ---
class PattentionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_agnostic_tokens, device='cpu'):
        super().__init__()
        self.scale = dim_in ** -0.5
        self.device = device

        # Task-Agnostic Parameters (always trainable)
        self.key_param_agnostic = nn.Parameter(torch.randn(num_agnostic_tokens, dim_in))
        self.value_param_agnostic = nn.Parameter(torch.randn(num_agnostic_tokens, dim_out))

        # Task-Specific Parameters (will grow over time)
        self.key_param_specific = nn.Parameter(torch.empty(0, dim_in, device=self.device))
        self.value_param_specific = nn.Parameter(torch.empty(0, dim_out, device=self.device))

        # Gradient mask for freezing old task-specific params
        self.register_buffer('specific_grad_mask', torch.empty(0, 1, device=self.device))
        self.growth_indices = [0] # Start with index 0 for the first task's params

    def forward(self, x):
        # Concatenate agnostic and specific parameters for the forward pass
        full_key_params = torch.cat([self.key_param_agnostic, self.key_param_specific], dim=0)
        full_value_params = torch.cat([self.value_param_agnostic, self.value_param_specific], dim=0)
        
        if full_key_params.shape[0] == 0:
            return torch.zeros(x.shape[:-1] + (full_value_params.shape[1],), device=self.device)

        similarity = torch.matmul(x, full_key_params.T)
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        
        out = torch.matmul(attn_weights, full_value_params)
        return out

    def grow(self, num_new_tokens):
        """ Grows the TASK-SPECIFIC parameter set. """
        print(f"  - Growing PattentionLayer with {num_new_tokens} new specific tokens.")
        dim_in = self.key_param_agnostic.shape[1]
        dim_out = self.value_param_agnostic.shape[1]

        # Freeze all existing specific parameters by setting their mask to 0
        if self.specific_grad_mask.numel() > 0:
            self.specific_grad_mask.fill_(0)

        # Create new parameters and a mask for them (they will be trainable)
        new_key_tokens = torch.randn(num_new_tokens, dim_in, device=self.device)
        new_value_tokens = torch.randn(num_new_tokens, dim_out, device=self.device)
        new_mask = torch.ones(num_new_tokens, 1, device=self.device)

        # Append new parameters and their masks
        self.key_param_specific = nn.Parameter(torch.cat([self.key_param_specific.data, new_key_tokens], dim=0))
        self.value_param_specific = nn.Parameter(torch.cat([self.value_param_specific.data, new_value_tokens], dim=0))
        self.specific_grad_mask = torch.cat([self.specific_grad_mask, new_mask], dim=0)
        
        self.growth_indices.append(self.key_param_specific.shape[0])

# --- FeedForward, Attention, and Encoder are updated to use the new PattentionLayer ---
# --- No major logic changes needed in them, they just instantiate the new layer ---
class TokenformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_agnostic_tokens, dropout = 0., device='cpu'):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pattn1 = PattentionLayer(dim, hidden_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pattn2 = PattentionLayer(hidden_dim, dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        x = self.layer_norm(x)
        x = self.pattn1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.pattn2(x)
        x = self.dropout2(x)
        return x + res

class TokenformerAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, num_agnostic_tokens=None, dropout = 0., device='cpu'):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        if num_agnostic_tokens is None:
            num_agnostic_tokens = dim
        self.to_q = PattentionLayer(dim, inner_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.to_k = PattentionLayer(dim, inner_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.to_v = PattentionLayer(dim, inner_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.to_out = PattentionLayer(inner_dim, dim, num_agnostic_tokens=num_agnostic_tokens, device=device) if project_out else nn.Identity()

    def forward(self, x):
        res = x
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)
        
        if q.shape[1] == 0: return torch.zeros_like(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) + res

class TokenformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_agnostic_tokens, dropout = 0., device='cpu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TokenformerAttention(dim, heads=heads, dim_head=dim_head, num_agnostic_tokens=num_agnostic_tokens, dropout=dropout, device=device),
                TokenformerFeedForward(dim, mlp_dim, num_agnostic_tokens=num_agnostic_tokens, dropout=dropout, device=device)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return self.norm(x)

class ContinualLearner(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, num_tasks, classes_per_task,
                 num_agnostic_tokens, device='cpu'):
        super().__init__()
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.device = device
        
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = create_feature_extractor(backbone, return_nodes={'layer4': 'features'})
        dim_backbone = 512
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapter = nn.Linear(dim_backbone, dim)

        self.growing_transformer = TokenformerEncoder(
            dim=dim, depth=depth, heads=heads, dim_head=dim, mlp_dim=mlp_dim,
            num_agnostic_tokens=num_agnostic_tokens, device=device
        )
        num_patches = 7 * 7
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_heads = nn.ModuleList([
            nn.Linear(dim, classes_per_task) for _ in range(num_tasks)
        ])

    def get_features(self, img):
        """Helper to get features for calibration and distillation."""
        self.backbone.eval()
        feature_map = self.backbone(img)['features']
        patch_embeddings = rearrange(feature_map, 'b d h w -> b (h w) d')
        adapted_embeddings = self.adapter(patch_embeddings)
        b, n, _ = adapted_embeddings.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        tokens = torch.cat((cls_tokens, adapted_embeddings), dim=1)
        tokens += self.pos_embedding
        output_sequence = self.growing_transformer(tokens)
        return output_sequence[:, 0]

    def forward(self, img, task_id, training=True):
        cls_output = self.get_features(img)
        
        if training:
            return self.mlp_heads[task_id](cls_output)
        else:
            # For evaluation, concatenate outputs from all heads seen so far
            outputs = [self.mlp_heads[i](cls_output) for i in range(task_id + 1)]
            return torch.cat(outputs, dim=1)

    def grow(self, num_new_specific_tokens):
        print("\n--- Growing Model (Adding Task-Specific Parameters) ---")
        for module in self.growing_transformer.modules():
            if isinstance(module, PattentionLayer):
                module.grow(num_new_specific_tokens)
        print("--- Model Growth Complete ---")