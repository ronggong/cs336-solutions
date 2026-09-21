import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, out)

def load_checkpoint(src, model, optimizer):
    d = torch.load(src)
    model.load_state_dict(d["model"])
    if optimizer is not None:
        optimizer.load_state_dict(d["optimizer"])
    return d["iteration"]