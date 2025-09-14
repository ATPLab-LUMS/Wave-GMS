# ------------------------------------------------------------------------------#
#
# File name                 : load_ckpt.py
# Purpose                   : Download/load checkpoints and build VAE/LiteVAE variants
# Usage                     : from tools.load_ckpt import get_tiny_autoencoder, get_sd_vae, get_lite_vae
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__                                 import annotations

import torch

from omegaconf                                  import OmegaConf

# Diffusers TinyVAE (HF weights)
from diffusers                                  import AutoencoderTiny as HF_TinyVAE

# Project imports
from modules.novel.lite_vae.encoder            import LiteVAEEncoder
from modules.novel.lite_vae.litevae            import LiteVAE
from modules.models.autoencoder                import AutoencoderKL
# ------------------------------------------------------------------------------#

# ------------------------------ Tiny VAE (HF) ---------------------------------#
def get_tiny_autoencoder(
    device: str | torch.device = "cpu",
    train : bool               = False,
    freeze: bool               = True,
    dtype : torch.dtype        = torch.float32,
):
    """
    Build AutoencoderTiny:
      - residual_autoencoding=False → load HF TinyVAE weights directly.
      - residual_autoencoding=True  → load HF weights, remap, then load into our Residual AutoencoderTiny().
    """
    print("[TinyVAE] Loading HF: madebyollin/taesd")
    vae = HF_TinyVAE.from_pretrained("madebyollin/taesd", torch_dtype=dtype, low_cpu_mem_usage=True)

    # Single move to target device/dtype
    vae = vae.to(device=device, dtype=dtype)
    if str(device).startswith("cuda"):
        vae = vae.to(memory_format=torch.channels_last)

    if train:
        print("[TinyVAE] Training mode")
        vae.train()
        return vae

    if freeze:
        print("[TinyVAE] Freezing parameters")
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()

    return vae
# ------------------------------------------------------------------------------#


# ------------------------------- LiteVAE (enc) --------------------------------#
def get_lite_vae(
    model_version: str                = "litevae-s",
    device       : str | torch.device = "cpu",
    dtype        : torch.dtype        = torch.float32,
    train        : bool               = True,
    freeze       : bool               = False,
) -> LiteVAE:
    """
    Build LiteVAE wrapper with LiteVAEEncoder (encoder → [mu|logvar] 8ch).
    """
    base_model = LiteVAE(LiteVAEEncoder(model_version=model_version))
    base_model = base_model.to(device=device, dtype=dtype)
    if str(device).startswith("cuda"):
        base_model = base_model.to(memory_format=torch.channels_last)

    if train:
        print("[LiteVAE] Training mode")
        base_model.train()
        return base_model

    if freeze:
        print("[LiteVAE] Freezing parameters")
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()

    return base_model
# ------------------------------------------------------------------------------#


# -------------------------------- SD-VAE (KL) ---------------------------------#
def get_sd_vae(
    ckpt_path : str,
    yaml_path : str,
    device    : str | torch.device = "cpu",
    dtype     : torch.dtype        = torch.float32,
    train     : bool               = False,
    freeze    : bool               = True,
):
    """
    Build Stable Diffusion first-stage AutoencoderKL from .ckpt + config YAML.

    Returns:
        (model, scale_factor)
    """
    # ---- Instantiate from YAML ----
    vae_config  = OmegaConf.load(yaml_path)
    params      = vae_config.first_stage_config.get("params", {})
    base_model  = AutoencoderKL(**params)

    # ---- Load checkpoint weights ----
    print(f"[SD-VAE] Loading checkpoint: {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd    = pl_sd.get("state_dict", pl_sd)  # handle both raw dict or {'state_dict': ...}
    missing, unexpected = base_model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[SD-VAE] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")

    scale_factor = vae_config.first_stage_config.get("scale_factor", 1.0)

    # ---- Device/dtype move (once) ----
    base_model = base_model.to(device=device, dtype=dtype)
    if str(device).startswith("cuda"):
        base_model = base_model.to(memory_format=torch.channels_last)

    # ---- Train / Freeze ----
    if freeze:
        print("[SD-VAE] Freezing parameters")
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()
    elif train:
        print("[SD-VAE] Training mode")
        base_model.train()

    return base_model, scale_factor
# ------------------------------------------------------------------------------#