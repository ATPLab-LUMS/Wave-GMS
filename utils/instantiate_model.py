# ------------------------------------------------------------------------------#
#
# File name                 : instantiate_model.py
# Purpose                   : Utility functions to instantiate AutoencoderTiny (Diffusers) 
#                             and LiteVAE (custom Wave-GMS variant).
# Usage                     : from utils.instantiate_model import get_tiny_autoencoder, get_lite_vae
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : Sep 16, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch

from diffusers                          import AutoencoderTiny
from modules.novel.lite_vae.encoder     import LiteVAEEncoder
from modules.novel.lite_vae.litevae     import LiteVAE
# ------------------------------------------------------------------------------#


# --------------------------- Tiny Autoencoder ---------------------------------#
def get_tiny_autoencoder(
    device: str     = "cuda" if torch.cuda.is_available() else "cpu",
    mode: str       = "tiny",
    train: bool     = False,
    freeze: bool    = True,
) -> AutoencoderTiny:
    """
    Load and configure AutoencoderTiny from HuggingFace Diffusers.
    Args:
        device  : target device (default = cuda if available).
        mode    : variant mode (currently only "tiny").
        train   : set to training mode.
        freeze  : freeze parameters for inference.
    """
    print(f"[INFO] Collecting AutoencoderTiny (mode={mode}) from Diffusers library...")
    vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd",
        torch_dtype=torch.float32,
        device_map=device,
    )

    if train:
        print("[INFO] AutoencoderTiny in training mode")
        return vae.train()

    if freeze:
        print("[INFO] Freezing AutoencoderTiny parameters...")
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
    return vae


# ------------------------------- LiteVAE --------------------------------------#
def get_lite_vae(
    model_version: str    = "litevae-s",
    device: str           = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype    = torch.float32,
    train: bool           = True,
    freeze: bool          = False,
) -> LiteVAE:
    """
    Instantiate LiteVAE model (Wave-GMS).
    Args:
        model_version : litevae variant ("litevae-s", "litevae-b", "litevae-m", "litevae-l").
        device        : target device.
        dtype         : tensor dtype.
        train         : set to training mode.
        freeze        : freeze parameters for inference.
    """
    model = (
        LiteVAE(LiteVAEEncoder(model_version=model_version))
        .to(device=device, dtype=dtype)
        .to(memory_format=torch.channels_last)
    )

    if train:
        print("[INFO] LiteVAE in training mode")
        return model.train()

    if freeze:
        print("[INFO] Freezing LiteVAE parameters...")
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    return model