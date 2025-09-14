# ------------------------------------------------------------------------------#
#
# File name                 : lightning.py
# Purpose                   : LightningModule wrappers for LMM and SFT-LMM (deep supervision)
# Usage                     : from networks.lmm.lightning import PL_LMM, PL_SFT_LMM
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 25, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch, logging
import torch.nn.functional          as F
import pytorch_lightning            as pl

from typing                         import Dict, Tuple

# ------------------------------------------------------------------------------#
#                         Helpers                                                #
# ------------------------------------------------------------------------------#
def _select_loss(name: str):
    name = (name or "l1").lower()
    if name == "l1":  return lambda p, t: F.l1_loss(p, t)
    if name == "l2":  return lambda p, t: F.mse_loss(p, t)
    raise ValueError(f"Unknown loss: {name}")

# ------------------------------------------------------------------------------#
#                         Lightning: LMM                                         #
# ------------------------------------------------------------------------------#
class PL_LMM(pl.LightningModule):
    """
    Core LMM Lightning wrapper.
    - core: ResAttnUNet_DS(z) -> dict(out, level1, level2, level3)
    - Deep supervision: weighted sum of losses vs target latent
    """
    def __init__(self, core, loss: str = "l1",
                 ds_weights: Tuple[float,float,float,float] = (1.0, 0.5, 0.25, 0.125),
                 lr: float = 1e-4, weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters(ignore=["core"])
        self.core           = core
        self.crit           = _select_loss(loss)
        self.ds_w           = ds_weights
        self.lr             = lr
        self.weight_decay   = weight_decay

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.core(z)

    def training_step(self, batch, _):
        z_in, z_tgt = batch["z_in"], batch["z_tgt"]               # (B,4,28,28)
        preds       = self(z_in)
        loss        = (
            self.ds_w[0] * self.crit(preds["out"],    z_tgt) +
            self.ds_w[1] * self.crit(preds["level1"], z_tgt) +
            self.ds_w[2] * self.crit(preds["level2"], z_tgt) +
            self.ds_w[3] * self.crit(preds["level3"], z_tgt)
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        z_in, z_tgt = batch["z_in"], batch["z_tgt"]
        preds       = self(z_in)
        loss        = F.l1_loss(preds["out"], z_tgt)
        self.log("val/l1_out", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        return opt

# ------------------------------------------------------------------------------#
#                         Lightning: SFT-LMM                                     #
# ------------------------------------------------------------------------------#
class PL_SFT_LMM(pl.LightningModule):
    """
    SFT-conditioned LMM Lightning wrapper.
    - core: SFT_UNet_DS(z, guidance) -> dict(...)
    - Expects batch to include "guidance" (already prepared).
    """
    def __init__(self, core, loss: str = "l1",
                 ds_weights: Tuple[float,float,float,float] = (1.0, 0.5, 0.25, 0.125),
                 lr: float = 1e-4, weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters(ignore=["core"])
        self.core           = core
        self.crit           = _select_loss(loss)
        self.ds_w           = ds_weights
        self.lr             = lr
        self.weight_decay   = weight_decay

    def forward(self, z: torch.Tensor, guidance: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.core(z, guidance)

    def training_step(self, batch, _):
        z_in, z_tgt, g = batch["z_in"], batch["z_tgt"], batch["guidance"]
        preds          = self(z_in, g)
        loss           = (
            self.ds_w[0] * self.crit(preds["out"],    z_tgt) +
            self.ds_w[1] * self.crit(preds["level1"], z_tgt) +
            self.ds_w[2] * self.crit(preds["level2"], z_tgt) +
            self.ds_w[3] * self.crit(preds["level3"], z_tgt)
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        z_in, z_tgt, g = batch["z_in"], batch["z_tgt"], batch["guidance"]
        preds          = self(z_in, g)
        loss           = F.l1_loss(preds["out"], z_tgt)
        self.log("val/l1_out", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        return opt