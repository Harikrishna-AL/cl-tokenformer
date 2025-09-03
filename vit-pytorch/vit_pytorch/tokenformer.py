# tokenformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from itertools import chain

# --- PattentionLayer and other Tokenformer modules are unchanged ---
class PattentionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_agnostic_tokens, device='cpu'):
        super().__init__()
        self.scale = dim_in ** -0.5
        self.device = device
        self.key_param_agnostic = nn.Parameter(torch.randn(num_agnostic_tokens, dim_in))
        self.value_param_agnostic = nn.Parameter(torch.randn(num_agnostic_tokens, dim_out))
        self.key_param_specific = nn.Parameter(torch.empty(0, dim_in, device=self.device))
        self.value_param_specific = nn.Parameter(torch.empty(0, dim_out, device=self.device))
        self.register_buffer('specific_grad_mask', torch.empty(0, 1, device=self.device))
        self.growth_indices = [0]
    def forward(self, x):
        full_key_params = torch.cat([self.key_param_agnostic, self.key_param_specific], dim=0)
        full_value_params = torch.cat([self.value_param_agnostic, self.value_param_specific], dim=0)
        if full_key_params.shape[0] == 0: return torch.zeros(x.shape[:-1] + (full_value_params.shape[1],), device=self.device)
        similarity = torch.matmul(x, full_key_params.T)
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        return torch.matmul(attn_weights, full_value_params)
    def grow(self, num_new_tokens):
        dim_in, dim_out = self.key_param_agnostic.shape[1], self.value_param_agnostic.shape[1]
        if self.specific_grad_mask.numel() > 0: self.specific_grad_mask.fill_(0)
        new_key_tokens = torch.randn(num_new_tokens, dim_in, device=self.device)
        new_value_tokens = torch.randn(num_new_tokens, dim_out, device=self.device)
        new_mask = torch.ones(num_new_tokens, 1, device=self.device)
        self.key_param_specific = nn.Parameter(torch.cat([self.key_param_specific.data, new_key_tokens], dim=0))
        self.value_param_specific = nn.Parameter(torch.cat([self.value_param_specific.data, new_value_tokens], dim=0))
        self.specific_grad_mask = torch.cat([self.specific_grad_mask, new_mask], dim=0)
        self.growth_indices.append(self.key_param_specific.shape[0])
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
        self.heads = heads; self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim); self.attend = nn.Softmax(dim = -1); self.dropout = nn.Dropout(dropout)
        if num_agnostic_tokens is None: num_agnostic_tokens = dim
        self.to_q = PattentionLayer(dim, inner_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.to_k = PattentionLayer(dim, inner_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.to_v = PattentionLayer(dim, inner_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.to_out = PattentionLayer(inner_dim, dim, num_agnostic_tokens=num_agnostic_tokens, device=device) if project_out else nn.Identity()
    def forward(self, x):
        res = x; x_norm = self.norm(x)
        q, k, v = self.to_q(x_norm), self.to_k(x_norm), self.to_v(x_norm)
        if q.shape[1] == 0: return torch.zeros_like(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots); attn = self.dropout(attn)
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
            x = attn(x); x = ff(x)
        return self.norm(x)

# --- MODIFIED: ContinualLearner with SSL components and parameter helpers ---
class ContinualLearner(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim,
                 num_tasks, classes_per_task, num_agnostic_tokens, channels=3, device='cpu'):
        super().__init__()
        self.num_tasks, self.classes_per_task, self.device = num_tasks, classes_per_task, device
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_height, self.patch_width = patch_height, patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.growing_transformer = TokenformerEncoder(dim=dim, depth=depth, heads=heads, dim_head=dim, mlp_dim=mlp_dim, num_agnostic_tokens=num_agnostic_tokens, device=device)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_heads = nn.ModuleList([nn.Linear(dim, classes_per_task) for _ in range(num_tasks)])
        
        # --- ADDED: SSL Projector and Predictor Heads (as per CaSSLe/TagFex) ---
        projection_dim = 128 # A smaller dimension for contrastive space is common
        self.contrastive_projector = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, projection_dim))
        self.contrastive_predictor = nn.Sequential(
            nn.Linear(projection_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, projection_dim))

    def agnostic_parameters(self):
        yield from self.to_patch_embedding.parameters()
        yield self.pos_embedding; yield self.cls_token
        for module in self.growing_transformer.modules():
            if isinstance(module, PattentionLayer):
                yield module.key_param_agnostic
                yield module.value_param_agnostic
    
    def ssl_parameters(self):
        yield from self.agnostic_parameters()
        yield from self.contrastive_projector.parameters()
        yield from self.contrastive_predictor.parameters()

    def supervised_parameters(self):
        agnostic_params = self.agnostic_parameters()
        specific_params = []
        for module in self.growing_transformer.modules():
            if isinstance(module, PattentionLayer) and module.key_param_specific.numel() > 0:
                trainable_indices = module.specific_grad_mask.view(-1).nonzero(as_tuple=True)[0]
                if len(trainable_indices) > 0:
                    # We need to yield the parameter object itself, not a slice
                    # The hook will handle the masking of gradients
                    specific_params.append(module.key_param_specific)
                    specific_params.append(module.value_param_specific)
        return chain(agnostic_params, iter(specific_params))

    def get_features(self, img, with_patch_tokens=False):
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
        patch_tokens_raw = x
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        output_sequence = self.growing_transformer(x)
        cls_output = output_sequence[:, 0]
        if with_patch_tokens: return cls_output, patch_tokens_raw
        return cls_output

    def forward(self, img_or_features, task_id, training=True):
        if img_or_features.dim() == 4: cls_output = self.get_features(img_or_features)
        else: cls_output = img_or_features
        if training: return self.mlp_heads[task_id](cls_output)
        else:
            outputs = [self.mlp_heads[i](cls_output) for i in range(task_id + 1)]
            return torch.cat(outputs, dim=1)

    def grow(self, num_new_specific_tokens):
        print("\n--- Growing Model (Adding Task-Specific Parameters) ---")
        for module in self.growing_transformer.modules():
            if isinstance(module, PattentionLayer): module.grow(num_new_specific_tokens)
        print("--- Model Growth Complete ---")