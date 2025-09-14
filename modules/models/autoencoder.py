# ------------------------------------------------------------------------------#
#
# File name                 : autoencoder.py
# Purpose                   : Implements KL-regularized autoencoder with optional
#                             Exponential Moving Average and discriminator loss.
# Usage                     : Imported by training script for latent space modeling.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import torch

import torch.nn.functional      as F
import pytorch_lightning        as pl

from contextlib     import contextmanager
from .model         import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution


# --------------------------- KL Autoencoder Model -----------------------------#
class AutoencoderKL(pl.LightningModule):
    """
    Lightning module implementing a VAE-style autoencoder with diagonal Gaussian
    latent space, optional EMA, and dual-phase loss (AE + discriminator).
    """

    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path        = None,
        ignore_keys      = [],
        image_key        = "image",
        colorize_nlabels = None,
        monitor          = None,
        ema_decay        = None,
        learn_logvar     = False
    ):
        super().__init__()

        self.image_key    = image_key
        self.learn_logvar = learn_logvar
        self.embed_dim    = embed_dim
        self.monitor      = monitor
        self.use_ema      = ema_decay is not None

        # Encoder / Decoder
        self.encoder         = Encoder(**ddconfig)
        self.decoder         = Decoder(**ddconfig)
        self.loss            = torch.nn.Identity()  # plug-in loss

        # Projection layers
        assert ddconfig["double_z"]
        self.quant_conv      = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # Optional colorizer for segmentation visualization
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        # Load pretrained if given
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    # --------------------------- Checkpoint loading ---------------------------#
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd   = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]

        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # --------------------------- EMA weight swap ------------------------------#
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context: print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context: print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # --------------------------- Encode / Decode ------------------------------#
    def encode(self, x):
        h        = self.encoder(x)
        moments  = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z    = self.post_quant_conv(z)
        dec  = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z         = posterior.sample() if sample_posterior else posterior.mode()
        dec       = self.decode(z)
        return dec, posterior

    # --------------------------- Batch input getter ---------------------------#
    def get_input(self, batch, key):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    # --------------------------- Training step logic --------------------------#
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs           = self.get_input(batch, self.image_key)
        recon, posterior = self(inputs)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(inputs, recon, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(inputs, recon, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc)
            return discloss

    # --------------------------- Validation step ------------------------------#
    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs           = self.get_input(batch, self.image_key)
        recon, posterior = self(inputs)

        aeloss, log_dict_ae     = self.loss(inputs, recon, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val" + postfix)
        discloss, log_dict_disc = self.loss(inputs, recon, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    # --------------------------- Optimizers -----------------------------------#
    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters())
        )

        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params.append(self.loss.logvar)

        opt_ae   = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # --------------------------- Image logging --------------------------------#
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        x   = self.get_input(batch, self.image_key).to(self.device)

        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                x    = self.to_rgb(x)
                xrec = self.to_rgb(xrec)

            log["samples"]        = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec

            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"]        = self.decode(torch.randn_like(posterior_ema.sample()))
                    log["reconstructions_ema"] = xrec_ema

        log["inputs"] = x
        return log

    def to_rgb(self, x):
        """
        Projects a segmentation map to RGB using a random linear projection.
        """
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


# --------------------------- Identity pass-through ----------------------------#
class IdentityFirstStage(torch.nn.Module):
    """
    A dummy module used when no autoencoder is needed.
    """

    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs): return x
    def decode(self, x, *args, **kwargs): return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs): return x

# --------------------------------- End -----------------------------------------#
