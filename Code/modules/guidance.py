# ------------------------------------------------------------------------------#
#
# File name                 : guidance.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

import torch, cv2, logging, contextlib, io

import numpy               as np
import torch.nn.functional as F
import torch.nn            as nn
from networks.novel.lite_vae.blocks.haar import HaarTransform

def get_edge_map(image: torch.Tensor, rgb = True) -> torch.Tensor:
    """
    Canny edge detection for input images.
    Args:
        image (B, 3, H, W) in range [0, 1]
    Returns:
        edge maps (B, 1, H, W)
    """
    edge_maps = []
    for img in image:
        img_np = img.permute(1, 2, 0).cpu().numpy() * 255
        img_gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        edge_tensor = torch.from_numpy(edges).float().unsqueeze(0) / 255.0
        edge_maps.append(edge_tensor)

    edge_map = torch.stack(edge_maps)
    if rgb:
        edge_map = edge_map.repeat(1, 3, 1, 1)
    return edge_map


def get_wavelet_subbands(images, lv = 3):
    wavelet_fn = HaarTransform()
    sub_bands  = wavelet_fn.dwt(images, level = lv) / (2 ** lv) # (B, 12, H / 2, W / 2)

    # Remove the approximation coefficient
    sub_bands_filter = sub_bands[:, 3:, :, :]

    return sub_bands_filter

def load_dino_silent():
    # Suppress logging from dinov2
    logging.getLogger().setLevel(logging.WARNING)

    # Suppress torch.hub stdout
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dino_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vits14',
            verbose=False
        )
    f = io.StringIO()
    return dino_model.to('cuda')
    # with contextlib.redirect_stdout(f):
    #     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    # return model

def get_dino_patch_features(image: torch.Tensor) -> torch.Tensor:
    """
    Extract patch-level features using a pretrained ViT (placeholder for DINO).
    Args:
        image (B, 3, H, W)
    Returns:
        patch features (B, C, h, w)
    """


    # dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino_model = load_dino_silent()
    dino_model.eval()  # Set to evaluation mode

    with torch.no_grad():
        result = dino_model(image, is_training=True)['x_norm_patchtokens']
        B, N, C = result.shape  # N = 28*28

        # Reshape to (B, C, H, W)
        h = w = int(N ** 0.5)
        dino_patch = result.reshape(B, h, w, C).permute(0, 3, 1, 2)

        # 🔹 Per-channel min–max normalization
        min_vals = dino_patch.amin(dim=(2, 3), keepdim=True)
        max_vals = dino_patch.amax(dim=(2, 3), keepdim=True)
        dino_patch = (dino_patch - min_vals) / (max_vals - min_vals + 1e-8)

    return dino_patch

        # result = dino_model(image, is_training = True)['x_norm_patchtokens'] # (B, 784, 384)

        # local_patch_feature = result.reshape(image.shape[0], int(math.sqrt(result.shape[1])), int(math.sqrt(result.shape[1])), 384) # --> reshaped to (B, 28, 28, 384)

        # return local_patch_feature.permute(0, 3, 1, 2)

def prepare_guidance(image: torch.Tensor, mode='edge') -> torch.Tensor:
    """
    Wrapper to select guidance type.
    """
    if mode == 'edge':
        return get_edge_map(image)
    elif mode == 'wavelet':
        return get_wavelet_subbands(image)
    elif mode == 'dino':
        return get_dino_patch_features(image)
    else:
        raise ValueError(f"Unknown guidance mode: {mode}")

class SKFF(nn.Module):
    def __init__(self, channels=3, reduction=8):
        super(SKFF, self).__init__()
        reduced_channels = max(1, channels // reduction)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fuse: shared compression
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.PReLU()
        )

        # Select: separate expansions
        self.expand_conv1 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.expand_conv2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.expand_conv3 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)

    def forward(self, wavelet_edges):
        """
        Input: wavelet_edges of shape (B, 3*C, H, W), where C is number of channels per stream (usually 3)
        """
        B, total_C, H, W = wavelet_edges.size()
        C = total_C // 3

        wavelet_h = wavelet_edges[:, 0: C, :, :]
        wavelet_v = wavelet_edges[:, C: 2 * C, :, :]
        wavelet_d = wavelet_edges[:, 2 * C: 3 * C, :, :]

        # Fuse step: global descriptor
        fused = wavelet_h + wavelet_v + wavelet_d
        gap = self.global_pool(fused)
        z = self.reduce_conv(gap)

        # Select step: generate attention weights
        v1 = self.expand_conv1(z)
        v2 = self.expand_conv2(z)
        v3 = self.expand_conv3(z)

        scores = torch.stack([v1, v2, v3], dim=1)  # (B, 3, C, 1, 1)
        weights = F.softmax(scores, dim=1)        # Softmax over streams

        s1, s2, s3 = weights[:, 0], weights[:, 1], weights[:, 2]
        out = s1 * wavelet_h + s2 * wavelet_v + s3 * wavelet_d  # Weighted sum

        return out  # (B, C, H, W)


# # Simulated batch of medical images (e.g., BUSI, GLaS) — normally (B, 3, 224, 224)
# B, C, H, W = 2, 3, 224, 224
# images = torch.rand(B, C, H, W)  # Values in [0, 1], float tensor

# # Example 1: Edge map guidance
# edge_guidance = prepare_guidance(images, mode='edge')
# print("Edge map shape:", edge_guidance.shape)  # (B, 1, H, W)

# # Example 2: Wavelet component
# wavelet_guidance = prepare_guidance(images, mode = 'wavelet')
# print(f'Wavelet shape:, {wavelet_guidance.shape}')

# # Example 3: DINO-style ViT patch features

# dino_guidance = prepare_guidance(images, mode = 'dino')
# print(f'Dino shape:, {dino_guidance.shape}')