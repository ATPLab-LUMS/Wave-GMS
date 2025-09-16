# ------------------------------------------------------------------------------#
#
# File name                 : utils.py
# Purpose                   : Utility functions for training/validation (checkpoints, seeding,
#                           learning rate, logging configs, etc.)
# Usage                     : from tools.utils import *
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

from __future__ import annotations

import os
import random
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor, optim


__all__ = [
    "mkdir",
    "seed_reproducer",
    "load_checkpoint",
    "save_checkpoint",
    "count_params",
    "adjust_learning_rate",
    "get_cuda",
    "print_options",
]


# --------------------------- File/Dir Utils ----------------------------------#
def mkdir(path: str) -> None:
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


# --------------------------- Reproducibility ---------------------------------#
def seed_reproducer(seed: int = 2333) -> None:
    """Fix random seeds for reproducibility across torch, numpy, python, cudnn."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.enabled       = True


# --------------------------- Checkpoints -------------------------------------#
def load_checkpoint(
    model: nn.Module,
    path: str,
    vae_model: Optional[nn.Module] = None,
    vae_model_load: bool = False,
    skff_model: Optional[nn.Module] = None,
    skff_model_load: bool = False,
) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]:
    """
    Load model/vae/skff states from checkpoint.
    """
    if not os.path.isfile(path):
        logging.warning(f"=> No checkpoint found at '{path}'")
        return None, vae_model, skff_model

    logging.info(f"=> Loading checkpoint '{path}'")
    state = torch.load(path, map_location="cpu", weights_only=True)

    model.load_state_dict(state["model"])
    logging.info("Loaded LMM model")

    if vae_model_load:
        if "vae_model" in state:
            vae_model.load_state_dict(state["vae_model"])
            logging.info("Loaded VAE model")
        else:
            logging.warning("VAE weights not found in checkpoint.")

    if skff_model_load:
        if "skff_model" in state:
            skff_model.load_state_dict(state["skff_model"])
            logging.info("Loaded SKFF model")
        else:
            logging.warning("SKFF weights not found in checkpoint.")

    return model, vae_model, skff_model


def save_checkpoint(
    model: nn.Module,
    save_name: str,
    path: str,
    vae_model: Optional[nn.Module] = None,
    vae_model_save: bool = False,
    skff_model: Optional[nn.Module] = None,
    skff_model_save: bool = False,
) -> None:
    """
    Save checkpoint with model/vae/skff states.
    """
    ckpt_dir = os.path.join(path, "checkpoints")
    mkdir(ckpt_dir)

    file_path = os.path.join(ckpt_dir, save_name)

    state = {"model": model.state_dict()}
    if vae_model_save and vae_model is not None:
        state["vae_model"] = vae_model.state_dict()
    if skff_model_save and skff_model is not None:
        state["skff_model"] = skff_model.state_dict()

    torch.save(state, file_path)

    logging.info(
        f"Checkpoint saved at {file_path} "
        f"(includes: model{' + vae' if vae_model_save else ''}{' + skff' if skff_model_save else ''})"
    )


# --------------------------- Training Utils ----------------------------------#
def count_params(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(
    optimizer: optim.Optimizer,
    initial_lr: float,
    epoch: int,
    reduce_epoch: int,
    decay: float = 0.5,
) -> float:
    """
    Step-decay LR schedule.
    """
    lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    logging.info(f"Learning rate adjusted to {lr:.6f}")
    return lr


def get_cuda(tensor: Tensor) -> Tensor:
    """Move tensor to CUDA if available."""
    return tensor.cuda() if torch.cuda.is_available() else tensor


def print_options(configs: Dict) -> None:
    """Pretty-print experiment configs and save to log file."""
    message = "----------------- Options ---------------\n"
    for k, v in configs.items():
        message += f"{k:>25}: {v:<30}\n"
    message += "----------------- End -------------------"
    logging.info(message)

    file_name = os.path.join(configs["log_path"], f"{configs['phase']}_configs.txt")
    with open(file_name, "wt") as f:
        f.write(message + "\n")