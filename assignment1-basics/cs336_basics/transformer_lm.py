import torch
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear
from .rope import RotaryPositionEmbedding

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, context_length, theta, num_layers, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionEmbedding(theta, d_model // num_heads, context_length, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype) for _ in range(num_layers)]
            )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, tokens_position: torch.Tensor):
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x, tokens_position)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x