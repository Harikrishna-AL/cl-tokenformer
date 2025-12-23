# tokenformer.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PattentionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_param_tokens, device='cpu'):
        super().__init__()
        self.device = device
        self.key_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_in))
        self.value_param_tokens = nn.Parameter(torch.randn(num_param_tokens, dim_out))
        self.register_buffer('key_grad_mask', torch.ones_like(self.key_param_tokens))
        self.register_buffer('value_grad_mask', torch.ones_like(self.value_param_tokens))
        
        # ### FIX ### Initialize split_idx
        self.split_idx = None

    def forward(self, x):
        similarity = torch.matmul(x, self.key_param_tokens.T) * (self.key_param_tokens.shape[1] ** -0.5)
        norm_similarity = F.normalize(similarity, p=2, dim=-1)
        attn_weights = F.gelu(norm_similarity)
        return torch.matmul(attn_weights, self.value_param_tokens)

    def grow(self, num_new_tokens):
        # ### FIX ### Correctly set the split_idx after growing
        num_old_tokens = self.key_param_tokens.shape[0]
        
        self.key_grad_mask.fill_(0)
        self.value_grad_mask.fill_(0)
        
        dim_in = self.key_param_tokens.shape[1]
        dim_out = self.value_param_tokens.shape[1]
        
        new_key_tokens = torch.randn(num_new_tokens, dim_in, device=self.device)
        new_value_tokens = torch.randn(num_new_tokens, dim_out, device=self.device)
        new_key_mask = torch.ones_like(new_key_tokens)
        new_value_mask = torch.ones_like(new_value_tokens)
        
        self.key_param_tokens = nn.Parameter(torch.cat([self.key_param_tokens.data, new_key_tokens], dim=0))
        self.value_param_tokens = nn.Parameter(torch.cat([self.value_param_tokens.data, new_value_tokens], dim=0))
        self.key_grad_mask = torch.cat([self.key_grad_mask, new_key_mask], dim=0)
        self.value_grad_mask = torch.cat([self.value_grad_mask, new_value_mask], dim=0)
        
        self.split_idx = num_old_tokens

class TokenformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_num_param_tokens, dropout=0., device='cpu'):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), PattentionLayer(dim, hidden_dim, num_param_tokens=ffn_num_param_tokens, device=device), nn.GELU(), nn.Dropout(dropout), PattentionLayer(hidden_dim, dim, num_param_tokens=dim, device=device), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class TokenformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, attn_num_param_tokens=None, dropout=0., device='cpu'):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads, self.scale = heads, dim_head ** -0.5
        self.norm, self.attend, self.dropout = nn.LayerNorm(dim), nn.Softmax(dim=-1), nn.Dropout(dropout)
        if attn_num_param_tokens is None: attn_num_param_tokens = inner_dim
        self.to_q, self.to_k, self.to_v = (PattentionLayer(dim, inner_dim, num_param_tokens=attn_num_param_tokens, device=device) for _ in range(3))
        self.to_out = PattentionLayer(inner_dim, dim, num_param_tokens=dim, device=device) if project_out else nn.Identity()

    def forward(self, x):
        x_norm = self.norm(x)
        q, k, v = self.to_q(x_norm), self.to_k(x_norm), self.to_v(x_norm)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)

class TokenformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, ffn_num_param_tokens, attn_num_param_tokens, dropout=0., device='cpu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([nn.ModuleList([
            TokenformerAttention(dim, heads=heads, dim_head=dim_head, attn_num_param_tokens=attn_num_param_tokens, dropout=dropout, device=device),
            TokenformerFeedForward(dim, mlp_dim, ffn_num_param_tokens=ffn_num_param_tokens, dropout=dropout, device=device)
        ]) for _ in range(depth)])
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TokenformerViT(nn.Module):
    def __init__(self, *, image_size, patch_size, classes_per_task, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., device='cpu'):
        super().__init__()
        self.device, self.dim, self.mlp_dim, self.classes_per_task = device, dim, mlp_dim, classes_per_task
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            PattentionLayer(patch_dim, dim, num_param_tokens=dim, device=device),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TokenformerEncoder(dim, depth, heads, dim_head, mlp_dim, mlp_dim, dim, dropout, device=device)
        self.pool, self.to_latent = pool, nn.Identity()
        self.classifier_head = PattentionLayer(dim, classes_per_task, num_param_tokens=classes_per_task * 4, device=device)

    def expand_classifier(self):
        print(f"🧠 Expanding classifier for {self.classes_per_task} new classes...")
        head = self.classifier_head
        num_param_tokens, old_num_classes = head.value_param_tokens.shape
        new_value_tokens = torch.randn(num_param_tokens, self.classes_per_task, device=self.device)
        head.value_param_tokens = nn.Parameter(torch.cat([head.value_param_tokens.data, new_value_tokens], dim=1))
        old_mask = torch.zeros(num_param_tokens, old_num_classes, device=self.device)
        new_mask = torch.ones(num_param_tokens, self.classes_per_task, device=self.device)
        head.value_grad_mask = torch.cat([old_mask, new_mask], dim=1)
        head.key_param_tokens.requires_grad = False
        print(f"✅ Classifier expanded. New output dimension: {head.value_param_tokens.shape[1]}")

    def grow_backbone(self, ffn_growth_factor=1, attn_growth_factor=1):
        print("\n--- 🧠 Growing Network Backbone ---")
        ffn_new_tokens, attn_new_tokens = self.mlp_dim // ffn_growth_factor, self.dim // attn_growth_factor
        for module in self.modules():
            if isinstance(module, PattentionLayer) and module is not self.classifier_head:
                is_ffn_layer = module.key_param_tokens.shape[1] != self.dim or module.value_param_tokens.shape[1] != self.dim
                num_new = ffn_new_tokens if is_ffn_layer else attn_new_tokens
                if num_new > 0: module.grow(num_new)
        print("--- ✅ Backbone Growth Complete ---")

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        features = self.to_latent(x)
        return self.classifier_head(features)