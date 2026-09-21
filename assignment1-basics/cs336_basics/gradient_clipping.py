import torch
import math

def gradient_clipping(params, max_gradient):
    global_norm = math.sqrt(math.fsum(torch.sum(p.grad**2) for p in params if p.grad is not None))
    if global_norm >= max_gradient:
        for p in params:
            if p.grad is None:
                continue
            p.grad *= (max_gradient / (global_norm + 10**(-6)))
    return global_norm