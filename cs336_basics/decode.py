import yaml
import torch
from pathlib import Path
from .transformer_lm import TransformerLM
from .bpe_tokenizer import Tokenizer
from .softmax import softmax
from .checkpointing import load_checkpoint

import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_file):
    """
    Sets up a logger with a custom format and outputs to both console and file.
    """
    logger = logging.getLogger("cs336")
    logger.setLevel(logging.DEBUG)  # Capture all levels: DEBUG and above

    # Prevent adding handlers multiple times if setup_logger() is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define log format
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Console shows INFO and above
    console_handler.setFormatter(log_format)

    # File handler with rotation (max 1MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=20_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # File logs everything
    file_handler.setFormatter(log_format)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

def greedy(y):
    return torch.argmax(y[:, -1, :], dim=-1)

def sampling(y):
    return torch.multinomial(y[:, -1, :], num_samples=1).squeeze(-1)

def topp(y, p=1.0):
    assert 0 < p <= 1
    if p == 1.0:
        return sampling(y)
    sorted, index = torch.sort(y[:, -1, :], dim=-1, descending=True)
    remove = torch.cumsum(sorted, dim=-1) > p
    removed = remove.clone()
    removed[..., 1:] = remove[..., :-1]
    removed[..., 0] = False
    a = sorted.masked_fill(removed, 0)
    b = a / torch.sum(a, dim=-1, keepdim=True)
    batch_row = torch.arange(0, sorted.size(0), device=y.device, dtype=torch.long)
    sample_sorted_column = torch.multinomial(b, num_samples=1).squeeze(-1)
    return index[batch_row, sample_sorted_column]

def decode(conf_yaml, checkpoint, vocab, merges, special_tokens, prompt, outdir):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    log = setup_logger(Path(outdir) / "log")
    log.info(f"cuda available={torch.cuda.is_available()}")

    with open(conf_yaml) as f:
        conf = yaml.safe_load(f)

    model_conf = conf["model"]
    decode_conf = conf["decode"]

    model = TransformerLM(vocab_size=model_conf["vocab_size"],
                          d_model=model_conf["d_model"],
                          num_heads=model_conf["num_heads"],
                          d_ff=model_conf["d_ff"],
                          context_length=model_conf["context_length"],
                          theta=model_conf["theta"],
                          num_layers=model_conf["num_layers"])

    device = None
    if model_conf["device"] == "cuda":
        device = torch.device("cuda")

    model.to(device=device)
    if model_conf["dtype"] == "bfloat16":
        model.to(dtype=torch.bfloat16)

    if checkpoint is not None:
        load_checkpoint(checkpoint, model, None)

    tokenizer = Tokenizer.from_files(vocab, merges, special_tokens)
    prompt_ids = tokenizer.encode(prompt)

    model.eval()
    with torch.no_grad():
        while True:
            x = torch.tensor(prompt_ids, device=device, dtype=torch.int).unsqueeze(dim=0)
            tokens_position = torch.arange(0, x.size(1), dtype=torch.int, device=device)
            logits = model(x, tokens_position)
            y = softmax(logits, -1, decode_conf["temperature"])
            if decode_conf["method"] == "greedy":
                pred = int(greedy(y)[0])
            elif decode_conf["method"] == "sampling":
                pred = int(sampling(y)[0])
            elif decode_conf["method"] == "top-p":
                pred = int(topp(y, decode_conf["p"])[0])
            else:
                raise ValueError(f"Invalid decode method {decode_conf['method']}")
            if tokenizer.decode([pred]) == "<|endoftext|>":
                break
            if len(prompt_ids) >= model_conf["context_length"]:
                break
            prompt_ids.append(pred)
    log.info(f"ids={prompt_ids}")
    log.info(f"text={tokenizer.decode(prompt_ids)}")