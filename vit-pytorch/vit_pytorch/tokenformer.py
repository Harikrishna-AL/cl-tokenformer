# tokenformer.py

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

# NOTE: PattentionLayer, TokenformerFeedForward, TokenformerAttention, 
#       and TokenformerEncoder are unchanged, except for the parameter
#       initialization in the PattentionLayer.grow() method.

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

    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        if self.key_param_tokens.shape[0] == 0:
            return torch.zeros(x.shape[:-1] + (self.value_param_tokens.shape[1],), device=self.device)

        similarity = torch.matmul(x, self.key_param_tokens.T) * self.scale
        if training and task_id > 0:
            bonus = torch.zeros_like(similarity)
            boundaries = [0] + self.growth_indices + [self.key_param_tokens.shape[0]]
            
            if task_id < len(boundaries) - 1:
                start_idx = boundaries[task_id]
                end_idx = boundaries[task_id + 1]
                if len(similarity.shape) == 3:
                    bonus[:, :, start_idx:end_idx] += attention_bonus
                else:
                    bonus[:, start_idx:end_idx] += attention_bonus
                similarity = similarity + bonus
            
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        self.attn_weights = attn_weights
        out = torch.matmul(attn_weights, self.value_param_tokens)
        return out

    def grow(self, num_new_tokens):
        dim_in = self.key_param_tokens.shape[1]
        dim_out = self.value_param_tokens.shape[1]
        num_old_tokens = self.key_param_tokens.shape[0]

        if num_old_tokens == 0:
            # Handle the very first growth if the layer starts empty
            new_key_tokens = torch.randn(num_new_tokens, dim_in, device=self.device)
            new_value_tokens = torch.randn(num_new_tokens, dim_out, device=self.device)
        else:
            # if not self.growth_indices: self.growth_indices.append(0)
            # --- NEW: Norm-Aligned Centroid Initialization ---
            # 1. Identify the most recently trained parameters
            if self.growth_indices:
                last_task_start_idx = self.growth_indices[-1] 
            else: 
                last_task_start_idx = 0

            old_keys_to_sample_from = self.key_param_tokens.data
            old_values_to_sample_from = self.value_param_tokens.data
            
            num_old_to_sample = old_keys_to_sample_from.shape[0]

            # 2. Randomly select indices from the old tokens (with replacement)
            random_indices = torch.randint(0, num_old_to_sample, (num_new_tokens,), device=self.device)

            # 3. Create the base for new tokens by sampling from the old ones
            base_new_keys = old_keys_to_sample_from[random_indices]
            base_new_values = old_values_to_sample_from[random_indices]

            # 4. Add small random noise to break symmetry and encourage specialization
            noise_k = torch.randn(num_new_tokens, dim_in, device=self.device) * 0.01
            noise_v = torch.randn(num_new_tokens, dim_out, device=self.device) * 0.01
            
            new_key_tokens = base_new_keys + noise_k
            new_value_tokens = base_new_values + noise_v

            # 5. (Optional but recommended) Align the norm to maintain energy
            avg_key_norm = torch.mean(torch.norm(old_keys_to_sample_from, p=2, dim=1))
            avg_value_norm = torch.mean(torch.norm(old_values_to_sample_from, p=2, dim=1))
            
            new_key_tokens = F.normalize(new_key_tokens, p=2, dim=1) * avg_key_norm
            new_value_tokens = F.normalize(new_value_tokens, p=2, dim=1) * avg_value_norm

        # Freeze all existing parameters by updating the mask
        self.key_grad_mask.fill_(0)
        self.value_grad_mask.fill_(0)
        
        # Create masks for the new tokens (these will be trainable)
        new_key_mask = torch.ones(num_new_tokens, dim_in, device=self.device)
        new_value_mask = torch.ones(num_new_tokens, dim_out, device=self.device)

        # print("old",self.key_param_tokens)
        # print("new",new_key_tokens)
        self.growth_indices.append(self.key_param_tokens.shape[0])
        # Append new parameters and masks
        self.key_param_tokens = nn.Parameter(torch.cat([self.key_param_tokens.data, new_key_tokens], dim=0))
        self.value_param_tokens = nn.Parameter(torch.cat([self.value_param_tokens.data, new_value_tokens], dim=0))
        self.key_grad_mask = torch.cat([self.key_grad_mask, new_key_mask], dim=0)
        self.value_grad_mask = torch.cat([self.value_grad_mask, new_value_mask], dim=0)
        
        # Record the start index of the *next* group of tokens
        


class TokenformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_num_param_tokens, dropout = 0., device='cpu', training=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pattn1 = PattentionLayer(dim, hidden_dim, num_param_tokens=ffn_num_param_tokens, device=device)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pattn2 = PattentionLayer(hidden_dim, dim, num_param_tokens=ffn_num_param_tokens, device=device)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        x_norm = self.layer_norm(x)
        x = self.pattn1(x_norm, task_id, attention_bonus, training)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.pattn2(x, task_id, attention_bonus, training)
        x = self.dropout2(x)
        return x

class TokenformerAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, attn_num_param_tokens=None, dropout = 0., device='cpu', training=True):
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
        self.to_out = PattentionLayer(inner_dim, dim, num_param_tokens=attn_num_param_tokens, device=device) if project_out else nn.Identity()

    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        x_norm = self.norm(x)
        q = self.to_q(x_norm, task_id, attention_bonus, training)
        k = self.to_k(x_norm, task_id, attention_bonus,training)
        v = self.to_v(x_norm, task_id, attention_bonus, training)
        if q.shape[1] == 0: return torch.zeros_like(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out, task_id, attention_bonus, training)

class TokenformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., device='cpu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TokenformerAttention(dim, heads=heads, dim_head=dim_head, attn_num_param_tokens=dim, dropout=dropout, device=device),
                TokenformerFeedForward(dim, mlp_dim, ffn_num_param_tokens=mlp_dim, dropout=dropout, device=device)
            ]))

    def forward(self, x, task_id=-1, attention_bonus=0.0, training=True):
        for attn, ff in self.layers:
            x = attn(x, task_id=task_id, attention_bonus=attention_bonus, training=training) + x
            x = ff(x, task_id=task_id, attention_bonus=attention_bonus, training=training) + x
        return self.norm(x)

class ContinualLearner(nn.Module):
    def __init__(self, *, image_size, dim, depth, heads, mlp_dim, num_tasks, classes_per_task,
                 channels=1, device='cpu', attention_bonus_max=0.0):
        super().__init__()
        self.attention_bonus_max = attention_bonus_max
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.device = device
        
        # Input dimension from flattened image
        input_dim = channels * image_size * image_size # e.g., 1 * 28 * 28 = 784 for MNIST

        # Projects the flattened image vector into the transformer's dimension
        self.input_projection = nn.Linear(input_dim, dim)

        self.growing_transformer = TokenformerEncoder(
            dim=dim, depth=depth, heads=heads, dim_head=dim, mlp_dim=mlp_dim, device=device
        )
        
        # No [CLS] token or positional embedding needed for a single-token sequence
        
        self.mlp_heads = nn.ModuleList([
            nn.Linear(dim, classes_per_task) for _ in range(num_tasks)
        ])
        
    def forward(self, img, task_id, training=True, current_attention_bonus=0.0):
        # 1. Flatten and project the input image to a single token
        img_flat = rearrange(img, 'b c h w -> b (c h w)')
        tokens = self.input_projection(img_flat).unsqueeze(1) # Shape: (b, 1, dim)

        # No [CLS] token or pos_embedding to add. The sequence length is 1.

        # 2. Pass the single-token sequence through the Tokenformer encoder
        output_sequence = self.growing_transformer(tokens, task_id, current_attention_bonus, training=training)
        
        # 3. Get the output token (it's the only one in the sequence) for classification
        output_token = output_sequence[:, 0] # Shape: (b, dim)

        # 4. Route to the appropriate head
        if training:
            # During training, we only need the output for the current task
            return self.mlp_heads[task_id](output_token)
        else:
            # --- MODIFIED: During evaluation, use the head for the given task_id ---
            # This is for a "task-aware" evaluation setting.
            outputs = []
            for t in range(self.num_tasks):
                out = self.mlp_heads[t](output_token)
                outputs.append(out)
            outputs = torch.cat(outputs, dim=1)
            # outputs = self.mlp_heads[task_id](output_token)
            return outputs
            # --- END MODIFICATION ---

    def grow(self):
        print("\n--- Growing Model (Tokenformer Encoder) ---")
        # Growth logic remains the same, growing parameters within PattentionLayers
        for module in self.growing_transformer.modules():
            if isinstance(module, PattentionLayer):
                if not module.growth_indices:
                    # For the first growth, grow by the initial number of tokens
                    # new_tokens_per_layer = module.key_param_tokens.shape[0]
                    new_tokens_per_layer = 32
                else:
                    # For subsequent growths, grow by the size of the first task's parameters
                    # new_tokens_per_layer = module.growth_indices[0]
                    new_tokens_per_layer = 32
                module.grow(new_tokens_per_layer)
        print("--- Model Growth Complete ---")