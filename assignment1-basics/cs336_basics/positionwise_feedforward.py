import torch
from .linear import Linear

class PWFF(torch.nn.Module):

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3 / 64) * 64

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.w1(x)
        silu = y * torch.sigmoid(y)

        linear = self.w3(x)

        return self.w2(silu * linear)

class SilUFF(torch.nn.Module):

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFNSiLU(𝑥) = 𝑊2 SiLU(𝑊1𝑥)
        
        x = self.w1(x)
        x = x * torch.sigmoid(x)
        return self.w2(x)