from torch.nn import Module, Parameter
from torch import Tensor
import torch
import math
from einops import einsum

class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        x = torch.empty([out_features, in_features], device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(x, 0, std, -3 * std, 3 * std)
        self.weight = Parameter(x)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")