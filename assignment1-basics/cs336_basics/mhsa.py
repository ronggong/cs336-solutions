import torch
from einops import einsum, rearrange
from .attention import scaled_dot_product_attention
from .linear import Linear

def calc_mask(x: torch.Tensor):
    return torch.tril(torch.ones(x.size(-2), x.size(-2), device=x.device, dtype=torch.bool))

class MHSA(torch.nn.Module):

    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_v = int(d_model / num_heads)
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def forward_qkv(self, x: torch.Tensor):
        qkv = einsum(x, torch.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0), 
                     "... seq_len d_model, d_out d_model -> ... seq_len d_out")
        q = rearrange(qkv[..., :self.d_model], "batch ... seq_len (num_heads d_k) -> batch num_heads ... seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        k = rearrange(qkv[..., self.d_model:-self.d_model], "batch ... seq_len (num_heads d_k) -> batch num_heads ... seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        v = rearrange(qkv[..., -self.d_model:], "batch ... seq_len (num_heads d_v) -> batch num_heads ... seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)
        return q, k, v

    def post_qkv(self, q, k, v, x):
        att = scaled_dot_product_attention(q, k, v, calc_mask(x))
        att = rearrange(att, "batch num_heads ... seq_len d_v -> batch ... seq_len (num_heads d_v)")
        return self.output_proj(att)

    def forward(self, x: torch.Tensor):
        q, k, v = self.forward_qkv(x)
        return self.post_qkv(q, k, v, x)

class MHSARoPE(MHSA):
    def __init__(self, d_model, num_heads, rope, device=None, dtype=None):
        super().__init__(d_model, num_heads, device, dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        q, k, v = self.forward_qkv(x)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        return self.post_qkv(q, k, v, x)
