import torch

def cross_entropy(o: torch.Tensor, x: int):
    o = o - torch.max(o, dim=-1, keepdim=True).values
    row = torch.arange(0, o.size(0), dtype=torch.int)
    l = torch.log(torch.sum(torch.exp(o), dim=-1)) - o[row, x]
    l = torch.mean(l, dim=0)
    return l
