import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        x = torch.empty([num_embeddings, embedding_dim], device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(x, 0, 1, -3, 3)
        self.weight = torch.nn.Parameter(x)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]