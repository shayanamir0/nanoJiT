import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

def apply_rope(x, freqs):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:, :x.shape[1]]
    x_out = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_out.type_as(x)

class JiTAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(dim // heads) 
        self.k_norm = RMSNorm(dim // heads) 
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), qkv)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, freqs_cis), apply_rope(k, freqs_cis)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        x = rearrange(attn @ v, 'b n h d -> b n (h d)')
        return self.proj(x)

class JiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = JiTAttention(dim, heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio * 2 / 3)
        self.mlp = SwiGLU(dim, hidden_dim, dropout=dropout) 
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, cond, freqs_cis):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1), freqs_cis)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        return x

class JustImageTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, dim=768, depth=12, heads=12, bottleneck=128, num_classes=10, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.ctx_start_block = 4 

        # bottleneck patch embedding 
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size*patch_size*3, bottleneck),
            nn.Linear(bottleneck, dim)
        )

        # time and class emmbedding (AdaLN)
        self.t_embedder = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.y_embedder = nn.Embedding(num_classes, dim)

        # In-Context class tokens
        self.ctx_len = 32
        self.ctx_class_emb = nn.Embedding(num_classes, dim) # lookup for the class token
        self.ctx_pos_emb = nn.Parameter(torch.randn(1, self.ctx_len, dim) * 0.02) # special pos embedding

        # RoPE frequencies
        head_dim = dim // heads
        self.register_buffer("freqs_cis", self.precompute_freqs_cis(head_dim, 4096))

        # backbone
        self.blocks = nn.ModuleList([JiTBlock(dim, heads, dropout=dropout) for _ in range(depth)])

        # final head
        self.final_norm = RMSNorm(dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_linear = nn.Linear(dim, patch_size*patch_size*3)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device='cpu')
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x, t, y): # Added y for class labels
        # embed inputs
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embed(x)

        # time + class for AdaLN
        cond = self.t_embedder(self.get_timestep_embedding(t).to(x.device)) + self.y_embedder(y)

        # prepare RoPE
        freqs_cis = self.freqs_cis[:x.shape[1] + self.ctx_len].to(x.device)

        for i, block in enumerate(self.blocks):
            # inject In-Context tokens at block 4 
            if i == self.ctx_start_block:
                ctx = self.ctx_class_emb(y).unsqueeze(1).repeat(1, self.ctx_len, 1)
                ctx = ctx + self.ctx_pos_emb
                x = torch.cat([ctx, x], dim=1) # prepend

            # slice RoPE for current sequence length
            block_freqs = freqs_cis[:x.shape[1]].view(1, x.shape[1], 1, -1)
            x = block(x, cond, block_freqs)

        # remove context tokens before output
        if len(self.blocks) > self.ctx_start_block:
            x = x[:, self.ctx_len:]

        shift, scale = self.final_adaLN(cond).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)
        return rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=int(x.shape[1]**0.5), p1=self.patch_size, p2=self.patch_size)

    def get_timestep_embedding(self, t, dim=256):
        half_dim = dim // 2
        freqs = torch.exp(-torch.arange(half_dim, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
