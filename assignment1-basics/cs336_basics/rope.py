import torch
from einops import rearrange

class RotaryPositionEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        x = rearrange(torch.linspace(0, max_seq_len-1, steps=max_seq_len, device=device, dtype=dtype), "seq_len -> seq_len 1")
        y = theta ** ((2*torch.linspace(1, int(d_k / 2), steps=int(d_k / 2), device=device, dtype=dtype) - 2) / d_k)
        sin = torch.sin(x / rearrange(y, "d_k -> 1 d_k"))
        cos = torch.cos(x / rearrange(y, "d_k -> 1 d_k"))
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        x_pair = rearrange(x, "... (pairs two) -> ... pairs two", two=2)
        a = x_pair[..., 0] * cos - x_pair[..., 1] * sin
        b = x_pair[..., 0] * sin + x_pair[..., 1] * cos
        return rearrange(torch.stack((a, b), dim=-1), "... pairs two -> ... (pairs two)", two=2)