# ------------------------------------------------------------------------------#
#
# File name                 : litevae.py
# Purpose                   : LiteVAE wrapper model (encoder + Gaussian latent)
# Usage                     : See example in `main()`
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# Note                      : Adapted from [https://arxiv.org/pdf/2405.14477.pdf]
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
from torch                   import nn, Tensor
from typing                  import Literal

from modules.novel.lite_vae.encoder import LiteVAEEncoder
from modules.novel.lite_vae.utils   import DiagonalGaussianDistribution
# ------------------------------------------------------------------------------#


# ------------------------------- LiteVAE Model --------------------------------#
class LiteVAE(nn.Module):
    """
    LiteVAE model:
      - Encoder (LiteVAEEncoder) with Haar-transform feature extraction
      - Latent distribution parameterized as [mu, logvar]
      - Gaussian sampling or deterministic mode
    """

    def __init__(
        self,
        encoder: LiteVAEEncoder = LiteVAEEncoder(),
        latent_dim: int = 4,
        use_1x1_conv: bool = False,
    ) -> None:
        super().__init__()

        self.encoder      = encoder
        self.wavelet_fn   = encoder.wavelet_fn

        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

        pre_channels      = latent_dim * 2  # [mu, logvar]
        self.pre_conv     = (
            nn.Conv2d(pre_channels, pre_channels, kernel_size=1)
            if use_1x1_conv else nn.Identity()
        )

    # --------------------------------------------------------------------------#
    def encode(self, image: Tensor) -> Tensor:
        """Encodes input image → [mu | logvar]."""
        return self.pre_conv(self.encoder(image))

    def forward(self, image: Tensor, sample: bool = True) -> Tensor:
        """
        Forward pass:
          - Encode image
          - Sample from Diagonal Gaussian (or take mean if sample=False)
        """
        latents_raw = self.encode(image).to(device=image.device, dtype=image.dtype)
        latent_dist = DiagonalGaussianDistribution(latents_raw)

        return latent_dist.sample() if sample else latent_dist.mode()


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    # Sanity check
    model  = LiteVAE()
    dummy  = torch.randn(2, 3, 224, 224)
    latent = model(dummy, sample=True)
    print(f"Latent shape: {latent.shape}")
    
    try:
        from torchinfo import summary
        summary(model, input_data=dummy, col_names=["input_size", "output_size", "num_params"])
    except ImportError:
        pass
    
# ------------------------------------------------------------------------------#   