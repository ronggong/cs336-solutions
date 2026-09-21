import torch
from collections.abc import Callable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.epsilon = eps
        self.lam = weight_decay

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data, requires_grad=False))
                v = state.get("v", torch.zeros_like(p.data, requires_grad=False))
                lr_t = lr * math.sqrt(1-self.beta2**t) / (1-self.beta1**t)
                p.data -= lr*self.lam*p.data
                state["m"] = self.beta1*m + (1-self.beta1)*p.grad.data
                state["v"] = self.beta2*v + (1-self.beta2)*p.grad.data**2
                p.data -= lr_t*state["m"] / (torch.sqrt(state["v"]) + self.epsilon)
                state["t"] = t + 1

        return loss