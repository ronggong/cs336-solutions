import torch
from .mhsa import MHSARoPE
from .positionwise_feedforward import PWFF
from .rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope, device=None, dtype=None):
        super().__init__()
        self.attn = MHSARoPE(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ffn = PWFF(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))