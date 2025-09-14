# ------------------------------------------------------------------------------#
#
# File name                 : train_utils.py
# Purpose                   : Utility functions for training/validation loops, e.g.
#                             multi-loss calculation, VAE helpers, etc.
# Usage                     : Imported by train.py and valid.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module imports ----------------------------------#
import torch, os, math

from networks          import *
from configs.config    import *
from PIL               import Image

# --------------------------- Multi-loss utility ------------------------------#
def get_multi_loss(
    criterion,
    out_dict,
    label,
    is_ds=True,
    key_list=None
):
    """
    Computes (optionally multi-scale) loss over outputs.
    Args:
        criterion  : loss function (e.g. MSE, DiceLoss)
        out_dict   : dict of outputs (multi-scale or single scale)
        label      : ground truth label tensor
        is_ds      : if True, sum over multiple keys; else just use 'out'
        key_list   : explicit list of keys to use (default: all keys in out_dict)
    Returns:
        multi_loss : scalar tensor (sum of all relevant losses)
    """
    keys = key_list if key_list is not None else list(out_dict.keys())
    if is_ds:
        multi_loss = sum([criterion(out_dict[key], label) for key in keys])
    else:
        multi_loss = criterion(out_dict["out"], label)
    return multi_loss

# ----------------------- VAE encoding helpers --------------------------------#
def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    """
    Extract mean and logvar from VAE posterior.
    Args:
        encoder_posterior : DiagonalGaussianDistribution object
        scale_factor      : scaling to apply to mean
    Returns:
        (mean, logvar)
    """
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(
            f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
        )
    return scale_factor * mean, logvar

# --------------------------- VAE decode utility ------------------------------#
def vae_decode(vae_model, pred_mean, scale_factor):
    """
    Decode latent to image, postprocess, and clamp.
    Args:
        vae_model   : model with .decode()
        pred_mean   : latent tensor
        scale_factor: latent scaling (float)
    Returns:
        pred_seg    : (B, 1, H, W) tensor, values in [0, 1]
    """
    z        = 1.0 / scale_factor * pred_mean
    pred_seg = vae_model.decode(z).sample  # shape: (B, C, H, W)
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True)  # (B, 1, H, W)
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0)
    return pred_seg

#-------------------------- Load Img for Infernece -----------------------------#
def load_img(path, img_size = 224, dtype_resize = np.float32):
    """Loads and normalizes a grayscale mask image to [0,1], resizes to (img_size, img_size)."""

    image = Image.open(path).convert("L").resize((img_size, img_size), resample=Image.NEAREST)
    image = np.array(image).astype(dtype_resize) / 255.0
    return image

# -------------------------- Save Binary and Logits ---------------------------#
def save_binary_and_logits(x_logits, x_binary, name, save_seg_img_path, save_seg_logits_path):
    """ Saves binary and logits images to specified path."""

    x_binary.save(os.path.join(save_seg_img_path, name + '_binary' + IMG_FORMAT))
    # Save x_logits as .png
    x_logits = (x_logits * 255).astype(np.uint8)
    x_logits = Image.fromarray(x_logits)
    x_logits.save(os.path.join(save_seg_logits_path, name + '_logits' + IMG_FORMAT))
# -------------------------------- End ----------------------------------------#
