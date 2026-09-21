import torch

def softmax(x: torch.Tensor, i: int, temp=1.0) -> torch.Tensor:
    x = x - torch.max(x, dim=i, keepdim=True).values
    y = torch.exp(x/temp)
    return y / torch.sum(y, dim=i, keepdim=True)