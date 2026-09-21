import argparse
import numpy as np
import yaml
import torch
from pathlib import Path
from einops import rearrange
from .data_loading import get_batch, get_batch_at_start
from .transformer_lm import TransformerLM
from .cross_entropy import cross_entropy
from .adamw import AdamW
from .checkpointing import load_checkpoint, save_checkpoint
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import learning_rate_schedule

import mlflow
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

def load_data(data_path):
    return np.memmap(data_path, dtype=np.uint16, mode="r")

def rearrange_logits(logits, y):
    return rearrange(logits, "B T V -> (B T) V"), rearrange(y, "B T -> (B T)")

def validation(model: torch.nn.Module, valid_set, batch_size, context_length, device):
    starts = range(0, len(valid_set)-context_length, context_length)
    model.eval()
    with torch.no_grad():
        loss_list = []
        for i in range(0, len(starts), batch_size):
            x, y = get_batch_at_start(valid_set, starts[i:i+batch_size], context_length, device=device)
            tokens_position = torch.arange(0, x.size(1), dtype=torch.int, device=device)
            logits = model(x, tokens_position)
            loss = cross_entropy(*rearrange_logits(logits, y))
            loss_list.append(float(loss))
        avg_loss = sum(loss_list) / len(loss_list)
    model.train()
    return avg_loss

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr

def train(train_data, valid_data, conf_yaml, outdir, checkpoint=None):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    log = setup_logger(Path(outdir) / "log")
    log.info(f"cuda available={torch.cuda.is_available()}")

    train_set = load_data(train_data)
    valid_set = load_data(valid_data)

    with open(conf_yaml) as f:
        conf = yaml.safe_load(f)

    model_conf = conf["model"]
    train_conf = conf["train"]
    data_conf = conf["data_loader"]
    optim_conf = conf["optimizer"]
    lr_conf = conf["lr_scheduler"]

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

    optimizer = AdamW(model.parameters(),
          lr=optim_conf["lr"],
          weight_decay=optim_conf["weight_decay"],
          betas=optim_conf["betas"],
          eps=optim_conf["eps"])

    if checkpoint is not None:
        load_checkpoint(Path(outdir) / checkpoint, model, optimizer)

    model.train()
    for i in range(train_conf["steps"]):
            lr = learning_rate_schedule(i,
                                amax=lr_conf["amax"],
                                amin=lr_conf["amin"],
                                tw=lr_conf["tw"],
                                tc=lr_conf["tc"])
            set_lr(optimizer, lr)

            x, y = get_batch(train_set, data_conf["batch_size"], data_conf["context_length"], device)
            tokens_position = torch.arange(0, x.size(1), dtype=torch.int, device=device)
            optimizer.zero_grad()
            logits = model(x, tokens_position)
            loss = cross_entropy(*rearrange_logits(logits, y))
            loss.backward()
            global_norm = gradient_clipping(model.parameters(), train_conf["grad_clip"])
            trn_loss = float(loss.detach())
            log.info(f"step={i}, lr={lr}, trn_loss={trn_loss}, grad_norm={global_norm}")
            mlflow.log_metric("trn_loss", trn_loss, step=i)
            optimizer.step()
            if i > 0 and i % train_conf["valid_steps"] == 0:
                loss_valid = validation(model, valid_set, data_conf["batch_size"], data_conf["context_length"], device)
                log.info(f"step={i}, valid_loss={loss_valid}")
                mlflow.log_metric("valid_loss", loss_valid, step=i)
            if i > 0 and i % train_conf["checkpointing_steps"] == 0:
                model_path = Path(outdir) / f"{i:07d}.pt"
                save_checkpoint(model, optimizer, i, model_path)