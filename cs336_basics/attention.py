import torch
import math
from einops import einsum
from .softmax import softmax

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask:torch.BoolTensor=None):
    qk = einsum(q, k, "... len_q d_k, ... len_k d_k -> ... len_q len_k")
    qk = qk / math.sqrt(k.size(-1))

    if mask is not None:
        qk.masked_fill_(~mask, float("-inf"))

    return einsum(softmax(qk, -1), v, "... len_q len_k, ... len_k d_v -> ... len_q d_v")

    