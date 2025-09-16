# ------------------------------------------------------------------------------#
#
# File name                 : decoder.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/decoder.py
# -----------------------------------------------------------------------------
# Utility to fetch a pretrained VAE from HuggingFace and expose **only** its
# decoder as a standalone nn.Module. Supports AutoencoderTiny (taesd) and
# AutoencoderKL (Stable‑Diffusion VAE).
# -----------------------------------------------------------------------------

import torch
from typing     import Literal
from __future__ import annotations
from torch      import nn

try:
    from diffusers import AutoencoderTiny, AutoencoderKL
except ImportError as e:  # diffusers not installed inside some unit‑test envs
    AutoencoderTiny = AutoencoderKL = None  # type: ignore

__all__ = ["load_pretrained_decoder"]


# -----------------------------------------------------------------------------
def _get_vae(model_type: Literal["tiny", "kl"], hf_id: str, dtype: torch.dtype) -> nn.Module:
    if model_type == "tiny":
        if AutoencoderTiny is None:
            raise ImportError("diffusers is required for AutoencoderTiny")
        return AutoencoderTiny.from_pretrained(hf_id, torch_dtype=dtype)
    elif model_type == "kl":
        if AutoencoderKL is None:
            raise ImportError("diffusers is required for AutoencoderKL")
        return AutoencoderKL.from_pretrained(hf_id, torch_dtype=dtype)
    else:
        raise ValueError("model_type must be 'tiny' or 'kl'")


def load_pretrained_decoder(
    model_type: Literal["tiny", "kl"] = "tiny",
    hf_model_id: str | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> nn.Module:
    """Return the **decoder** sub‑module from a pretrained VAE.

    Parameters
    ----------
    model_type : {'tiny', 'kl'}
        Which diffusers VAE architecture to pull.
    hf_model_id : str | None
        HuggingFace model ID. Defaults:
          • tiny → "madebyollin/taesd"
          • kl   → "stabilityai/sd‑vae‑ft‑mse"
    dtype : torch.dtype
        Precision to load weights in.
    device : str | torch.device
        Device for the decoder.
    freeze : bool, default = True
        If True, sets `param.requires_grad = False` and `.eval()`.
    """

    if hf_model_id is None:
        hf_model_id = "madebyollin/taesd" if model_type == "tiny" else "stabilityai/sd‑vae‑ft‑mse"

    vae = _get_vae(model_type, hf_model_id, dtype)
    decoder = vae.decoder.to(device)
    decoder.load_state_dict(vae.decoder.state_dict())  # deep copy to detach from VAE graph

    if freeze:
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False

    return decoder