# tokenformer.py

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from einops import rearrange, repeat

# PattentionLayer, TokenformerFeedForward, TokenformerAttention, TokenformerEncoder
# are unchanged. They are the core building blocks.
class PattentionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_param_tokens, device='cpu'):
        super().__init__()
        self.scale = dim_in ** -0.5
        self.device = device
        if num_param_tokens == 0:
            self.key_param_tokens = nn.Parameter(torch.empty(0, dim_in))
            self.value_param_tokens = nn.Parameter(torch.empty(0, dim_out))
        else:
            self.key_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_in))
            self.value_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_out))
        self.register_buffer('key_grad_mask', torch.ones_like(self.key_param_tokens))
        self.register_buffer('value_grad_mask', torch.ones_like(self.value_param_tokens))
        self.growth_indices = []
        self.attn_weights = None

    def forward(self, x, task_id=-1, attention_bonus=0.0):
        if self.key_param_tokens.shape[0] == 0:
            # ... (empty layer handling is the same) ...
            return torch.zeros(x.shape[:-1] + (self.value_param_tokens.shape[1],), device=self.device)

        similarity = torch.matmul(x, self.key_param_tokens.T) * self.scale
        
        # ### NEW: Apply the Attention Bonus during Training ###
        if self.training and task_id != -1:
            bonus = torch.zeros_like(similarity)
            boundaries = [0] + self.growth_indices + [self.key_param_tokens.shape[0]]
            
            if task_id < len(boundaries) - 1:
                start_idx = boundaries[task_id]
                end_idx = boundaries[task_id + 1]
                
                # Add the bonus only to the similarity scores of the current task's params
                if len(similarity.shape) == 3: # For sequence data (B, N, D)
                    bonus[:, :, start_idx:end_idx] += attention_bonus
                else: # For non-sequence data (B, D)
                    bonus[:, start_idx:end_idx] += attention_bonus
                
                similarity = similarity + bonus
            
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        self.attn_weights = attn_weights
        out = torch.matmul(attn_weights, self.value_param_tokens)
        return out

    def grow(self, num_new_tokens):
        # ... (grow method is unchanged) ...
        dim_in = self.key_param_tokens.shape[1]
        dim_out = self.value_param_tokens.shape[1]
        num_old_tokens = self.key_param_tokens.shape[0]
        if num_old_tokens > 0:
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

### --- Modules updated to pass task_id --- ###
class TokenformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_num_param_tokens, dropout = 0., device='cpu'):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pattn1 = PattentionLayer(dim, hidden_dim, num_param_tokens=ffn_num_param_tokens, device=device)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pattn2 = PattentionLayer(hidden_dim, dim, num_param_tokens=ffn_num_param_tokens, device=device)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, task_id=-1, attention_bonus=0.0):
        x_norm = self.layer_norm(x)
        x = self.pattn1(x_norm, task_id, attention_bonus)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.pattn2(x, task_id, attention_bonus)
        x = self.dropout2(x)
        return x

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
        self.to_q = PattentionLayer(dim, inner_dim, num_param_tokens=attn_num_param_tokens, device=device)
        self.to_k = PattentionLayer(dim, inner_dim, num_param_tokens=attn_num_param_tokens, device=device)
        self.to_v = PattentionLayer(dim, inner_dim, num_param_tokens=attn_num_param_tokens, device=device)
        self.to_out = PattentionLayer(inner_dim, dim, num_param_tokens=attn_num_param_tokens, device=device) if project_out else nn.Identity()

    def forward(self, x, task_id=-1, attention_bonus=0.0):
        x_norm = self.norm(x)
        q = self.to_q(x_norm, task_id, attention_bonus)
        k = self.to_k(x_norm, task_id, attention_bonus)
        v = self.to_v(x_norm, task_id, attention_bonus)
        # ... (rest of the forward pass is the same) ...
        if q.shape[1] == 0: return torch.zeros_like(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out, task_id, attention_bonus)

class TokenformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., device='cpu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TokenformerAttention(dim, heads=heads, dim_head=dim_head, attn_num_param_tokens=0, dropout=dropout, device=device),
                TokenformerFeedForward(dim, mlp_dim, ffn_num_param_tokens=0, dropout=dropout, device=device)
            ]))

    def forward(self, x, task_id=-1, attention_bonus=0.0):
        for attn, ff in self.layers:
            x = attn(x, task_id, attention_bonus) + x
            x = ff(x, task_id, attention_bonus) + x
        return self.norm(x)

class ContinualLearner(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, num_tasks, classes_per_task,
                #  attn_tokens_per_task, ffn_tokens_per_task, 
                 device='cpu', attention_bonus=0.0): # Add bonus here
        super().__init__()
        self.attention_bonus = attention_bonus # Store bonus value
        # ... (the rest of __init__ is the same as the "fair" version) ...
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.device = device
        # self.attn_tokens_per_task = attn_tokens_per_task
        # self.ffn_tokens_per_task = ffn_tokens_per_task
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = create_feature_extractor(backbone, return_nodes={'layer4': 'features'})
        dim_backbone = 512
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.adapter = nn.Linear(dim_backbone, dim)
        self.growing_transformer = TokenformerEncoder(
            dim=dim, depth=depth, heads=heads, dim_head=dim, mlp_dim=mlp_dim, device=device
        )
        num_patches = 7 * 7
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_heads = nn.ModuleList([
            nn.Linear(dim, classes_per_task) for _ in range(num_tasks)
        ])
        
    def forward(self, img, task_id):
        self.backbone.eval()
        feature_map = self.backbone(img)['features']
        patch_embeddings = rearrange(feature_map, 'b d h w -> b (h w) d')
        adapted_embeddings = self.adapter(patch_embeddings)
        b, n, _ = adapted_embeddings.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        tokens = torch.cat((cls_tokens, adapted_embeddings), dim=1)
        tokens += self.pos_embedding
        
        # Pass the task_id AND bonus down to the encoder during training
        output_sequence = self.growing_transformer(tokens, task_id, self.attention_bonus)
        cls_output = output_sequence[:, 0]
        
        return self.mlp_heads[task_id](cls_output)
    def grow(self):
        print("\n--- Growing Model (Tokenformer Encoder) ---")
        # NOTE: Growth factor is hardcoded for simplicity
        # new_tokens_per_layer = 128 // 2 
        for module in self.growing_transformer.modules():
            if isinstance(module, PattentionLayer):
                if module.growth_indices == []:
                    new_tokens_per_layer = module.key_param_tokens.shape[0]
                else:
                    new_tokens_per_layer = module.growth_indices[0]
                module.grow(new_tokens_per_layer)
        print("--- Model Growth Complete ---")