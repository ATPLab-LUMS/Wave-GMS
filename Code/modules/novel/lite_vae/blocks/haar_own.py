# ------------------------------------------------------------------------------#
#
# File name                 : haar_own.py
# Purpose                   : Albumentations-based dataset with Excel/default splits
#                             and robust RGB image/mask loading for GMS pipelines.
# Usage                     : from data.image_dataset import ImageDataset
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

import torch, pywt, re

import numpy             as np
import matplotlib.pyplot as plt

from torch  import nn, Tensor
from typing import List, Tuple, Dict
from typing import Optional

class HaarTransform(nn.Module):
    """
    Recursive 2D Haar Wavelet Transform (DWT + iDWT) with optional visualization.

    Supports PyTorch (B, C, H, W) tensors, and can output stacked or nested coeffs.
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels

    # ------------------------------------------------------------------
    def dwt_recursive(self, x: np.ndarray) -> List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Perform recursive Haar DWT and return list of [(LL, (LH, HL, HH))] from shallow to deep.
        """
        coeffs_all = []
        current = x
        for _ in range(self.levels):
            coeffs2 = pywt.dwt2(current, "haar")
            LL, (LH, HL, HH) = coeffs2
            coeffs_all.append((LL, (LH, HL, HH)))
            current = LL
        return coeffs_all

    # ------------------------------------------------------------------
    def reconstruct_from_coeffs(
        self, coeffs_all: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ) -> np.ndarray:
        """
        Reconstruct full image using pywt.waverec2 from coeffs list.
        """
        LL_final, _ = coeffs_all[-1]
        detail_list = [t[1] for t in reversed(coeffs_all)]  # deep → shallow
        coeffs = [LL_final, *detail_list]
        return pywt.waverec2(coeffs, "haar")

    # ------------------------------------------------------------------
    def plot_wavelet_batch(
        self,
        level_tensors: List[Tensor],
        *,
        originals: Optional[Tensor] = None,
        reconstructed: Optional[Tensor] = None,
        cmap: str = "gray"
    ):
        """
        Visualize multi-level wavelet sub-bands alongside originals & reconstructions.

        Parameters
        ----------
        level_tensors : List[Tensor]
            • len(level_tensors)  = num_levels
            • level_tensors[lvl]  = Tensor (B, 4, H_l, W_l)
        
        originals : Tensor (B, C, H, W), optional
            Original images (only first channel shown)
        
        reconstructed : Tensor (B, C, H, W), optional
            Reconstructed images (only first channel shown)

        cmap : str
            Matplotlib colormap for grayscale display.
        """
        originals     = originals[:, 0: 1] if originals is not None else None
        reconstructed = reconstructed[:, 0: 1] if reconstructed is not None else None

        num_levels = len(level_tensors)
        batch_size = level_tensors[0].size(0)
        show_recon = reconstructed is not None
        show_orig  = originals is not None

        total_cols = num_levels * 4 + show_orig + show_recon

        fig, axes = plt.subplots(
            nrows=batch_size,
            ncols=total_cols,
            figsize=(3.5 * total_cols, 2.5 * batch_size),
            squeeze=False
        )

        titles = ["LL", "LH", "HL", "HH"]

        for b in range(batch_size):
            col = 0

            # Show Original
            if show_orig:
                ax = axes[b, col]
                img = originals[b, 0].cpu().numpy()
                ax.imshow(img, cmap=cmap)
                if b == 0:
                    ax.set_title("Original")
                ax.axis("off")
                col += 1

            # Subbands
            for lvl, tensor in enumerate(level_tensors, 1):
                for sub in range(4):
                    band = tensor[b, sub].cpu().numpy()
                    ax = axes[b, col]
                    ax.imshow(band, cmap=cmap)
                    if b == 0:
                        ax.set_title(f"{titles[sub]}-{lvl}")
                    ax.axis("off")
                    col += 1

            # Show Reconstruction
            if show_recon:
                ax = axes[b, col]
                img = reconstructed[b, 0].cpu().numpy()
                ax.imshow(img, cmap=cmap)
                if b == 0:
                    ax.set_title("Reconstructed")
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def group_by_level_strict(self, 
        coeff_dict: Dict[str, np.ndarray],
        *,
        batch_size: int,
        num_levels: int,
        return_dict = True
    ) -> List[torch.Tensor] | Dict[str, torch.Tensor]:
        """
        Convert the dict from HaarTransform(stacked_version=True) into a list of
        tensors [level0, level1, level2] with shape (B, 4, H, W).

        Guarantees:
        • Each (batch_i, level_j) appears exactly once.
        • Tensors are ordered so batch dimension is aligned across levels.
        • Raises ValueError if any coefficient is missing or duplicated.
        """
        
        _key_re = re.compile(r"batch_(\d+)_level_(\d+)")
         
        # Allocate empty grid: grid[level][batch] = array
        grid: List[List[np.ndarray | None]] = [
            [None] * batch_size for _ in range(num_levels)
        ]
        
        # Fill the grid
        for key, arr in coeff_dict.items():
            m = _key_re.fullmatch(key)
            if m is None:
                raise ValueError(f"Key '{key}' does not match 'batch_i_level_j'")
            b_idx, l_idx = map(int, m.groups())
            if not (0 <= b_idx < batch_size) or not (0 <= l_idx < num_levels):
                raise ValueError(f"Key '{key}' index out of range")
            if grid[l_idx][b_idx] is not None:
                raise ValueError(f"Duplicate entry for batch {b_idx} level {l_idx}")
            grid[l_idx][b_idx] = arr

        # Ensure no missing entries
        for l, row in enumerate(grid):
            for b, arr in enumerate(row):
                if arr is None:
                    raise ValueError(f"Missing coeffs for batch {b} level {l}")

        # Stack per level into tensors
        level_tensors: List[torch.Tensor] = []
        for row in grid:                                # iterate over levels
            stacked = torch.stack([torch.from_numpy(arr) for arr in row], dim=0)
            level_tensors.append(stacked)

        if return_dict:
            return level_tensors, {f"level_{idx}": tensor for idx, tensor in enumerate(level_tensors)}
        else:
            return level_tensors

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor | list,
        *,
        inverse: bool = False,
        nchw: bool = True,
        stacked_version: bool = True,
        visualize: bool = False
    ):
        """
        Perform Haar DWT or inverse DWT.
        """
        batch_size, c, h, w = x.shape
        coeffs_all = self.dwt_recursive(x)
        # coeffs_batch = []
        # for b in range(batch_size):
        #     img_coeffs = []
        #     for LL, (LH, HL, HH) in coeffs_all:
        #         LL_b  = LL[b]   # shape (C, H, W)
        #         LH_b  = LH[b]
        #         HL_b  = HL[b]
        #         HH_b  = HH[b]
        #         img_coeffs.append((LL_b, (LH_b, HL_b, HH_b)))
        #     coeffs_batch.append(img_coeffs)
        coeffs_batch = []

        # Convert to numpy outside the loop to prevent slow memory migration
        x_np = x[:, 0].detach().cpu().numpy().astype(np.float32)

        for i in range(batch_size):
            coeffs_i = self.dwt_recursive(x_np[i])  # grayscale input --> [(LL, (LH, HL, HH)), ] list
            coeffs_batch.append(coeffs_i)

        if inverse:
            recon = torch.from_numpy(self.reconstruct_from_coeffs(coeffs_all)).float()

        if not nchw:
            raise NotImplementedError("Only NCHW tensors are supported.")

        if stacked_version:
            # Build dict in one go using nested comprehension
            dict_coeffs = {
                f"batch_{bi}_level_{li}": np.stack([LL, LH, HL, HH], axis=0)
                for bi, coeffs_i in enumerate(coeffs_batch)
                for li, (LL, (LH, HL, HH)) in enumerate(coeffs_i)
            }
            level_tensors_lst, level_tensors_dict = self.group_by_level_strict(dict_coeffs, batch_size= batch_size, num_levels=self.levels)
            if visualize:
                self.plot_wavelet_batch(level_tensors_lst, originals = x if isinstance(x, Tensor) else None, reconstructed = recon if inverse else None)
            if inverse:
                return recon, level_tensors_lst, level_tensors_dict
            else:
                return level_tensors_lst, level_tensors_dict
        return coeffs_batch