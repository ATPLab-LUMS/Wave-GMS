# ------------------------------------------------------------------------------#
#
# File name                 : distributions.py
# Purpose                   : Implements Gaussian and Dirac distributions for VAE-style
#                             latent space modeling, including KL and NLL computations.
# Usage                     : Used by AutoencoderKL to sample, compute KL, and NLL.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import torch
import numpy as np


# --------------------------- Abstract interface -------------------------------#
class AbstractDistribution:
    """
    Base class for latent distributions. Requires sample() and mode().
    """
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


# --------------------------- Deterministic wrapper ----------------------------#
class DiracDistribution(AbstractDistribution):
    """
    A no-variance (delta function) distribution.
    Returns the same tensor for both sample and mode.
    """
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


# --------------------------- Gaussian distribution ----------------------------#
class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian latent distribution with log-variance.

    Args:
        parameters    : Tensor of shape (B, 2C, H, W), containing [mean | logvar]
        deterministic : If True, variance is zero (Dirac-like behavior)
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters    = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        self.logvar        = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        """
        Reparameterized sampling from the distribution.
        """
        device = self.parameters.device
        eps    = torch.randn(self.mean.shape).to(device)
        return self.mean + self.std * eps

    def mode(self):
        return self.mean

    def mu_and_sigma(self):
        return self.mean, self.logvar

    def kl(self, other=None):
        """
        KL divergence to standard normal (if other is None) or another Gaussian.
        """
        if self.deterministic:
            return torch.tensor([0.], device=self.parameters.device)

        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3]
            )
        else:
            return 0.5 * torch.sum(
                ((self.mean - other.mean) ** 2) / other.var +
                self.var / other.var -
                1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3]
            )

    def nll(self, sample, dims=[1, 2, 3]):
        """
        Negative log-likelihood for reconstruction loss.
        """
        if self.deterministic:
            return torch.tensor([0.], device=self.parameters.device)

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + ((sample - self.mean) ** 2) / self.var,
            dim=dims
        )


# --------------------------- Manual KL computation ----------------------------#
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute KL divergence between two diagonal Gaussians:
    KL[N1 || N2] where N1 ~ (mean1, logvar1), N2 ~ (mean2, logvar2)

    Source:
        https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/losses.py
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "At least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

# --------------------------------- End -----------------------------------------#
