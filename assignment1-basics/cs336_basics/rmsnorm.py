import torch
from einops import rearrange

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        x = torch.ones([d_model], device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps)

        #g = rearrange(self.g, "d_model -> 1 1 d_model")

        result = x * self.weight / rms

        return result.to(in_dtype)