# ------------------------------------------------------------------------------#
#
# File name                 : __init__.py
# Purpose                   : Model registry and builders (LMM + SFT-LMM)
# Usage                     : from networks import build_model
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 25, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
from utils.registry           import Registry

from modules.latent_mapping_model    import ResAttnUNet_DS
from modules.sft_lmm                 import SFT_UNet_DS
from modules.novel.lite_vae.encoder  import LiteVAEEncoder
from modules.novel.lite_vae.litevae  import LiteVAE

# Lightning wrappers
from modules.lmm.lightning           import PL_LMM, PL_SFT_LMM

MODEL_REGISTRY                      = Registry("MODEL")

# ------------------------------------------------------------------------------#
#                         Plain nn.Module builders                              #
# ------------------------------------------------------------------------------#
@MODEL_REGISTRY.register("lmm")
def build_lmm(*, in_ch=4, out_ch=4, ch=32, ch_mult=(1,2,4,4)):
    return ResAttnUNet_DS(in_channel=in_ch, out_channels=out_ch, ch=ch, ch_mult=ch_mult)

@MODEL_REGISTRY.register("sft_lmm")
def build_sft_lmm(*, in_ch=4, out_ch=4, ch=32, ch_mult=(1,2,4,4), guidance_ch=64):
    return SFT_UNet_DS(in_channels=in_ch, out_channels=out_ch, ch=ch, ch_mult=ch_mult, guidance_channels=guidance_ch)

MODEL_REGISTRY = Registry("MODEL")

@MODEL_REGISTRY.register("litevae_encoder")
def build_litevae_encoder(*, version="litevae-b", in_ch=12, out_ch=12):
    return LiteVAEEncoder(model_version=version, in_channels=in_ch, out_channels=out_ch)

@MODEL_REGISTRY.register("litevae_encoder_only")
def build_litevae_encoder_only(*, version="litevae-b"):
    # encoder→moments(8ch)→DiagonalGaussianDistribution will happen in your pipeline (or via LiteVAE)
    enc = LiteVAEEncoder(model_version=version, in_channels=12, out_channels=12)
    return enc

@MODEL_REGISTRY.register("litevae")   # encoder + optional decoder hook
def build_litevae(*, version="litevae-b", latent_dim=4, decode=False, use_1x1_conv=False):
    enc = LiteVAEEncoder(model_version=version, in_channels=12, out_channels=12)
    return LiteVAE(encoder=enc, decoder=None, latent_dim=latent_dim, output_type="image",
                   use_1x1_conv=use_1x1_conv, decode=decode)

def build_model(name: str, **kwargs):
    return MODEL_REGISTRY.get(name)(**kwargs)

# ------------------------------------------------------------------------------#
#                         LightningModule builders                                #
# ------------------------------------------------------------------------------#
@MODEL_REGISTRY.register("lmm_pl")
def build_lmm_pl(*, in_ch=4, out_ch=4, ch=32, ch_mult=(1,2,4,4),
                 loss="l1", ds_weights=(1.0, 0.5, 0.25, 0.125), lr=1e-4, wd=0.0):
    core = build_lmm(in_ch=in_ch, out_ch=out_ch, ch=ch, ch_mult=ch_mult)
    return PL_LMM(core, loss=loss, ds_weights=ds_weights, lr=lr, weight_decay=wd)

@MODEL_REGISTRY.register("sft_lmm_pl")
def build_sft_lmm_pl(*, in_ch=4, out_ch=4, ch=32, ch_mult=(1,2,4,4), guidance_ch=64,
                     loss="l1", ds_weights=(1.0, 0.5, 0.25, 0.125), lr=1e-4, wd=0.0):
    core = build_sft_lmm(in_ch=in_ch, out_ch=out_ch, ch=ch, ch_mult=ch_mult, guidance_ch=guidance_ch)
    return PL_SFT_LMM(core, loss=loss, ds_weights=ds_weights, lr=lr, weight_decay=wd)

# ------------------------------------------------------------------------------#
#                         Public API                                             #
# ------------------------------------------------------------------------------#
def build_model(name: str, **kwargs):
    return MODEL_REGISTRY.get(name)(**kwargs)