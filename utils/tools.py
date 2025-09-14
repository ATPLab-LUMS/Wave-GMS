# ------------------------------------------------------------------------------#
#
# File name                 : tools.py
# Purpose                   : Utility functions for seeding, saving/loading models,
#                             learning rate scheduling, CUDA setup, config loading.
# Usage                     : Used across training and evaluation pipelines.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 21, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import os, yaml, torch, random, logging

import numpy as np

from pathlib import Path
from datetime import datetime


# --------------------------- Create directories -------------------------------#
def mkdir(path: str):
    """
    Create a directory if it doesn't already exist.

    Args:
        path : Directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


# --------------------------- Seed reproducibility -----------------------------#
def seed_reproducer(seed: int = 2333):
    """
    Set all seeds for reproducibility (NumPy, random, torch, and CUDA).

    Args:
        seed : Random seed to set (default = 2333).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


# --------------------------- Load model checkpoint ----------------------------#
def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Load a model checkpoint from a given path.

    Args:
        model : Model instance to load weights into.
        path  : Path to .pth checkpoint file.

    Returns:
        model : Updated model or None if file not found.
    """
    if os.path.isfile(path):
        logging.info("=> loading checkpoint '{}'".format(path))

        # Map checkpoint to CPU (avoid GPU mismatch)
        state = torch.load(
            path, weights_only=True, map_location=lambda storage, location: storage
        )

        # Load model weights
        model.load_state_dict(state["model"], strict = True)
        logging.info("Loaded")
        
    else:
        model = None
        logging.info("=> no checkpoint found at '{}'".format(path))
        
    return model


# --------------------------- Save model checkpoint ----------------------------#
def save_checkpoint(model: torch.nn.Module, save_name: str, path: str) -> None:
    """
    Save model weights to disk.

    Args:
        model     : PyTorch model instance.
        save_name : Name for saved .pth file.
        path      : Directory to save in (checkpoints subfolder will be created).
    """
    model_savepath = os.path.join(path, "checkpoints")
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)

    file_name = os.path.join(model_savepath, save_name)

    torch.save({"model": model.state_dict()}, file_name)
    logging.info("save model to {}".format(file_name))


# --------------------------- Adjust learning rate -----------------------------#
def adjust_learning_rate(optimizer, initial_lr, epoch, reduce_epoch, decay=0.5):
    """
    Step decay learning rate scheduler.

    Args:
        optimizer     : PyTorch optimizer.
        initial_lr    : Starting learning rate.
        epoch         : Current epoch.
        reduce_epoch  : Epoch frequency to decay.
        decay         : Multiplicative decay factor.

    Returns:
        lr : Updated learning rate.
    """
    lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    logging.info("Change Learning Rate to {}".format(lr))
    return lr


# --------------------------- Move tensor to CUDA ------------------------------#
def get_cuda(tensor):
    """
    Moves a tensor to CUDA if available.

    Args:
        tensor : A PyTorch tensor.

    Returns:
        tensor : Tensor on GPU if available.
    """
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


# --------------------------- Print configuration ------------------------------#
def print_options(config):
    """
    Pretty print and save the current configuration (supports .py config as module or dict).
    """
    # Accept both module or dict config
    if not isinstance(config, dict):
        config = {k: getattr(config, k) for k in dir(config) if k.isupper() and not k.startswith('__')}
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in config.items():
        message += "{:>25}: {:<30}\n".format(str(k), str(v))
    message += "----------------- End -------------------"

    logging.info(message)

    # Save options to file
    log_path  = config.get('LOG_PATH', './logs')
    phase     = config.get('PHASE', 'train')
    file_name = os.path.join(log_path, f"{phase}_configs.txt")
    
    with open(file_name, "wt") as opt_file:
        opt_file.write(message)
        opt_file.write("\n")

# --------------------------- Precision helper ---------------------------------#
def get_precision_dtypes(prec: str = "float16"):
    """
    Maps precision string → (numpy dtype, torch dtype).

    Args:
        prec : "float32", "float16", "bfloat16", …

    Returns:
        Tuple[np.dtype, torch.dtype]
    """
    table = {
        "float32": (np.float32, torch.float32),
        "float16": (np.float16, torch.float16),
        "bfloat16": (np.float32, torch.bfloat16),  # Albumentations needs fp32 array
    }
    if prec not in table:
        raise ValueError(f"Unsupported precision '{prec}'.")
    return table[prec]


# --------------------------------- End -----------------------------------------#
