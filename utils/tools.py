# ------------------------------------------------------------------------------#
#
# File name                 : checkpoint_utils.py
# Purpose                   : Utility functions for checkpoint handling, 
#                             reproducibility, logging, and training support.
# Usage                     : from utils.checkpoint_utils import load_checkpoint, save_checkpoint
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : Sep 17, 2025
# Note                      : Adapted from [https://github.com/King-HAW/GMS]
# ------------------------------------------------------------------------------#

# ---------------------------------- Module Imports ----------------------------#
import os, random, logging, torch

import numpy as np
# ------------------------------------------------------------------------------#


# --------------------------- Filesystem Helpers -------------------------------#
def mkdir(path: str) -> None:
    """Create directory if it doesn’t exist."""
    if not os.path.exists(path):
        os.makedirs(path)


# --------------------------- Seed Reproducer ----------------------------------#
def seed_reproducer(seed: int = 2333) -> None:
    """Reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.enabled       = True


# --------------------------- Checkpoint Loading -------------------------------#
def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    vae_model=None,
    vae_model_load: bool = False,
    skff_model=None,
    skff_model_load: bool = False,
):
    """Load checkpoint and optionally load VAE/SKFF models if present."""
    if os.path.isfile(path):
        logging.info(f"=> Loading checkpoint '{path}'")
        state = torch.load(path, weights_only=True, map_location="cpu")

        model.load_state_dict(state["model"])
        logging.info(f"Loaded LMM model from {path}")

        if vae_model_load:
            if "vae_model" in state:
                vae_model.load_state_dict(state["vae_model"])
                logging.info(f"Loaded VAE model from {path}")
            else:
                logging.warning("VAE model not found in checkpoint.")

        if skff_model_load:
            if "skff_model" in state:
                skff_model.load_state_dict(state["skff_model"])
                logging.info(f"Loaded SKFF model from {path}")
            else:
                logging.warning("SKFF model not found in checkpoint.")
    else:
        logging.info(f"=> No checkpoint found at '{path}'")
        model = None

    return model, vae_model, skff_model


# --------------------------- Checkpoint Saving --------------------------------#
def save_checkpoint(
    model: torch.nn.Module,
    save_name: str,
    path: str,
    vae_model=None,
    vae_model_save: bool = False,
    skff_model=None,
    skff_model_save: bool = False,
) -> None:
    """Save model checkpoint with optional VAE and SKFF components."""
    model_savepath = os.path.join(path, "checkpoints")
    mkdir(model_savepath)

    file_name = os.path.join(model_savepath, save_name)
    save_dict = {"model": model.state_dict()}

    if vae_model_save:
        save_dict["vae_model"] = vae_model.state_dict()
        logging.info(f"Saving LMM model and VAE model to {file_name}")

    if skff_model_save:
        save_dict["skff_model"] = skff_model.state_dict()
        logging.info(f"Saving LMM model and SKFF model to {file_name}")

    torch.save(save_dict, file_name)
    logging.info(f"Saved checkpoint to {file_name}")


# --------------------------- Model Utilities ----------------------------------#
def count_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(optimizer, initial_lr, epoch, reduce_epoch, decay: float = 0.5) -> float:
    """Decay learning rate exponentially after fixed epochs."""
    lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    logging.info(f"Changed learning rate to {lr}")
    return lr


def get_cuda(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to CUDA if available."""
    return tensor.cuda() if torch.cuda.is_available() else tensor


# --------------------------- Config Logging -----------------------------------#
def print_options(configs: dict) -> None:
    """Print and save configuration options."""
    message = "----------------- Options ---------------\n"
    for k, v in configs.items():
        message += f"{k:>25}: {v:<30}\n"
    message += "----------------- End -------------------"
    logging.info(message)

    file_name = os.path.join(configs["log_path"], f"{configs['phase']}_configs.txt")
    with open(file_name, "wt") as opt_file:
        opt_file.write(message + "\n")
# ------------------------------------------------------------------------------#