# tokenformer.py

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange, repeat

# --- NEW: Separation Autoencoder ---
class SeparationAutoencoder(nn.Module):
    """
    An autoencoder that learns to create a separated feature representation.
    - The encoder is the "separation layer".
    - The decoder is used to enforce a reconstruction loss, ensuring the
      encoded features remain informative.
    """
    def __init__(self, input_dim, hidden_dim, separated_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, separated_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(separated_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        separated_features = self.encoder(x)
        reconstructed_features = self.decoder(separated_features)
        return separated_features, reconstructed_features

# PattentionLayer, TokenformerFeedForward, etc. are unchanged
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
    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        if self.key_param_tokens.shape[0] == 0: return torch.zeros(x.shape[:-1] + (self.value_param_tokens.shape[1],), device=self.device)
        similarity = torch.matmul(x, self.key_param_tokens.T) * self.scale
         ### EDITED: Added this line back for visualization purposes ###
        self.attn_weights = similarity 
        attn_weights = F.gelu(F.normalize(similarity, p=2, dim=-1))
        return torch.matmul(attn_weights, self.value_param_tokens)
    def grow(self, num_new_tokens):
        dim_in, dim_out = self.key_param_tokens.shape[1], self.value_param_tokens.shape[1]
        num_old_tokens = self.key_param_tokens.shape[0]
        if num_old_tokens > 0: self.growth_indices.append(num_old_tokens)
        self.key_grad_mask.fill_(0); self.value_grad_mask.fill_(0)
        new_key_tokens, new_value_tokens = torch.randn(num_new_tokens, dim_in, device=self.device) * 0.01, torch.randn(num_new_tokens, dim_out, device=self.device) * 0.01
        self.key_param_tokens = nn.Parameter(torch.cat([self.key_param_tokens.data, new_key_tokens], dim=0))
        self.value_param_tokens = nn.Parameter(torch.cat([self.value_param_tokens.data, new_value_tokens], dim=0))
        self.key_grad_mask = torch.cat([self.key_grad_mask, torch.ones_like(new_key_tokens)], dim=0)
        self.value_grad_mask = torch.cat([self.value_grad_mask, torch.ones_like(new_value_tokens)], dim=0)
class TokenformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_num_param_tokens, dropout = 0., device='cpu', training=True):
        super().__init__()
        self.layer_norm, self.pattn1, self.gelu, self.dropout1, self.pattn2, self.dropout2 = nn.LayerNorm(dim), PattentionLayer(dim, hidden_dim, num_param_tokens=ffn_num_param_tokens, device=device), nn.GELU(), nn.Dropout(dropout), PattentionLayer(hidden_dim, dim, num_param_tokens=ffn_num_param_tokens, device=device), nn.Dropout(dropout)
    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        x_norm = self.layer_norm(x)
        x = self.pattn1(x_norm, task_id, attention_bonus, training)
        x = self.dropout1(self.gelu(x))
        x = self.dropout2(self.pattn2(x, task_id, attention_bonus, training))
        return x
class TokenformerAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, attn_num_param_tokens=None, dropout = 0., device='cpu', training=True):
        super().__init__()
        inner_dim, project_out = dim_head * heads, not (heads == 1 and dim_head == dim)
        self.heads, self.scale, self.norm, self.attend, self.dropout = heads, dim_head ** -0.5, nn.LayerNorm(dim), nn.Softmax(dim = -1), nn.Dropout(dropout)
        if attn_num_param_tokens is None: attn_num_param_tokens = inner_dim
        self.to_q, self.to_k, self.to_v = PattentionLayer(dim, inner_dim, attn_num_param_tokens, device=device), PattentionLayer(dim, inner_dim, attn_num_param_tokens, device=device), PattentionLayer(dim, inner_dim, attn_num_param_tokens, device=device)
        self.to_out = PattentionLayer(inner_dim, dim, attn_num_param_tokens, device=device) if project_out else nn.Identity()
    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        x_norm = self.norm(x)
        q, k, v = self.to_q(x_norm, task_id, attention_bonus, training), self.to_k(x_norm, task_id, attention_bonus,training), self.to_v(x_norm, task_id, attention_bonus, training)
        if q.shape[1] == 0: return torch.zeros_like(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out, task_id, attention_bonus, training)
class TokenformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., device='cpu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([nn.ModuleList([TokenformerAttention(dim, heads, dim_head, dim, dropout, device), TokenformerFeedForward(dim, mlp_dim, mlp_dim, dropout, device)]) for _ in range(depth)])
    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        for attn, ff in self.layers:
            x = attn(x, task_id, attention_bonus, training) + x
            x = ff(x, task_id, attention_bonus, training) + x
        return self.norm(x)

class ContinualLearner(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, num_tasks, classes_per_task, device='cpu'):
        super().__init__()
        self.num_tasks, self.classes_per_task, self.device = num_tasks, classes_per_task, device
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = create_feature_extractor(backbone, return_nodes={'layer4': 'features'})
        dim_backbone = 512
        for param in self.backbone.parameters(): param.requires_grad = False
        
        # --- MODIFIED: Replaced ProjectionHead with SeparationAutoencoder ---
        self.separation_autoencoder = SeparationAutoencoder(
            input_dim=dim_backbone,
            hidden_dim=dim_backbone,
            separated_dim=dim # The separated dimension matches Tokenformer's dimension
        )

        self.growing_transformer = TokenformerEncoder(dim=dim, depth=depth, heads=heads, dim_head=dim, mlp_dim=mlp_dim, device=device)
        self.pos_embedding = nn.Parameter(torch.randn(1, 7 * 7 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_heads = nn.ModuleList([nn.Linear(dim, classes_per_task) for _ in range(num_tasks)])

    def separation_layer_params(self):
        return self.separation_autoencoder.parameters()

    def continual_learning_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'separation_autoencoder' not in name:
                yield param
        
    def forward(self, img, task_id, training=True):
        self.backbone.eval()
        resnet_features_map = self.backbone(img)['features']
        resnet_features_sequence = rearrange(resnet_features_map, 'b d h w -> b (h w) d')

        # Pass ResNet features through the new autoencoder
        separated_features_sequence, reconstructed_features_sequence = self.separation_autoencoder(resnet_features_sequence)
        separated_feature_vector = torch.mean(separated_features_sequence, dim=1)
        
        b, n, _ = separated_features_sequence.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        tokens = torch.cat((cls_tokens, separated_features_sequence), dim=1)
        tokens += self.pos_embedding
        
        output_sequence = self.growing_transformer(tokens, task_id, training=training)
        cls_output = output_sequence[:, 0]
        
        output = self.mlp_heads[task_id](cls_output)
        
        # Return all necessary outputs for the different training phases
        return output, resnet_features_sequence, separated_feature_vector, reconstructed_features_sequence

    def grow(self):
        print("\n--- Growing Model (Tokenformer Encoder) ---")
        for module in self.growing_transformer.modules():
            if isinstance(module, PattentionLayer):
                new_tokens = module.key_param_tokens.shape[0] // (len(module.growth_indices) + 1) if module.growth_indices else module.key_param_tokens.shape[0]
                module.grow(new_tokens)
        print("--- Model Growth Complete ---")