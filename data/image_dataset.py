# ------------------------------------------------------------------------------#
#
# File name                 : image_dataset.py
# Purpose                   : Albumentations-based dataset with Excel/default splits
#                             and robust RGB image/mask loading for GMS pipelines.
# Usage                     : from data.image_dataset import ImageDataset
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__ import annotations

import os, random, logging, pickle

from pathlib                 import Path
from typing                  import Dict, List, Tuple, Optional, Literal

import numpy                 as np
import pandas                as pd
from PIL                     import Image

import albumentations        as A
from albumentations.pytorch  import ToTensorV2

import torch
from torch.utils.data        import Dataset
# ------------------------------------------------------------------------------#

__all__ = [
    "generate_pickle_default",
    "ImageDataset",
]

# --------------------------- Split helpers ------------------------------------#


def generate_pickle_default(
    root_dir: str | Path,
    split: float = 0.9,
    name: str = "kvasir-seg_train_test_names.pkl",
    img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    mask_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    seed: int = 42,
) -> Path:
    """
    Create a simple train/test pickle by scanning:
        root_dir/
            images/
            masks/
    It matches images & masks by basename and splits by ratio.

    Returns
    -------
    Path to the generated pickle file at `root_dir/name`.
    """
    root     = Path(root_dir)
    img_dir  = root / "images"
    mask_dir = root / "masks"

    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected folders: {img_dir} and {mask_dir}")

    def list_with_exts(folder: Path, exts: Tuple[str, ...]) -> List[str]:
        return [p.name for p in folder.iterdir() if p.suffix.lower() in exts]

    img_files   = list_with_exts(img_dir,  img_exts)
    mask_files  = list_with_exts(mask_dir, mask_exts)
    img_bases   = {Path(f).stem for f in img_files}
    mask_bases  = {Path(f).stem for f in mask_files}
    common      = sorted(list(img_bases & mask_bases))
    
    if not common:
        raise RuntimeError(f"No matched image-mask basenames found under {root}")

    random.seed(seed)
    random.shuffle(common)

    train_size  = int(split * len(common))
    train_bases = common[:train_size]
    test_bases  = common[train_size:]

    pickle_dict = {"train": {"name_list": train_bases}, "test": {"name_list": test_bases}}
    pkl_path    = root / name
    with open(pkl_path, "wb") as f:
        pickle.dump(pickle_dict, f)

    logging.info(f"[splits] saved → {pkl_path} | train={len(train_bases)} test={len(test_bases)}")
    return pkl_path


# --------------------------- Dataset ------------------------------------------#
class ImageDataset(Dataset):
    """
    Albumentations-based dataset with **RGB masks** (preserved as 3-channel).

    Modes
    -----
    - excel=True:
        Expects Excel split files in: Dataset/<root_dir>/
        Images/Masks live in the *same* folder as the pickle (combined 'images'/'masks').
        Mask filenames have prefix 'mask_'.
    - excel=False:
        Expects combined folders already created:
            Dataset/<root_dir>/{images,masks}
        Builds/uses a simple train/test pickle next to those folders.

    Returns
    -------
    dict: {"name": str, "img": Tensor(3,H,W), "seg": Tensor(3,H,W)}
    """

    def __init__(
        self,
        pickle_file_path: Optional[str | Path] = None,
        root_dir: str | Path = "QaTar-19",
        stage: Literal["train", "val", "test", "whole"] = "train",
        excel: bool = False,
        img_size: int = 224,
        img_ext: str = ".png",
        mask_ext: str = ".png",
        transforms_train: Optional[A.Compose] = None,
        transforms_eval: Optional[A.Compose]  = None,
    ) -> None:
        super().__init__()

        # Resolve or create split pickle
        if pickle_file_path is None:
            dataset_root = Path("Dataset") / root_dir
            pickle_file_path = generate_pickle_default(dataset_root)

        with open(pickle_file_path, "rb") as f:
            split = pickle.load(f)

        base_dir         = Path(pickle_file_path).parent
        self.img_dir     = base_dir / "images"
        self.mask_dir    = base_dir / "masks"
        self.stage       = stage
        self.img_size    = img_size
        self.img_ext     = img_ext
        self.mask_ext    = mask_ext
        self.excel       = excel

        if stage == "whole":
            name_list = split.get("train", {}).get("name_list", []) + split.get("test", {}).get("name_list", [])
            name_list += split.get("val", {}).get("name_list", [])
        else:
            if stage not in split:
                raise KeyError(f"Stage '{stage}' not found in split pickle.")
            name_list = split[stage]["name_list"]

        self.name_list = list(sorted(name_list))
        self.transforms_train = transforms_train or self._default_train_transforms()
        self.transforms_eval  = transforms_eval  or self._default_eval_transforms()
        logging.info(f"[dataset] {stage:>5s}: {len(self.name_list):5d} @ {self.img_size}x{self.img_size}")

        if not self.img_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected folders next to pickle: {self.img_dir} and {self.mask_dir}")

    # ------------------------------------------------------------------ #
    def _default_train_transforms(self) -> A.Compose:
        return A.Compose(
            [
                A.ToFloat(max_value=255.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=20, p=0.4),
                A.Resize(self.img_size, self.img_size, interpolation=1),  # bilinear
                ToTensorV2(),
            ]
        )

    def _default_eval_transforms(self) -> A.Compose:
        return A.Compose(
            [
                A.ToFloat(max_value=255.0),
                A.Resize(self.img_size, self.img_size, interpolation=1),
                ToTensorV2(),
            ]
        )

    # ------------------------------------------------------------------ #
    def _load_rgb(self, path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

    def _img_path(self, name: str) -> Path:
        return self.img_dir / f"{name}{self.img_ext}"

    def _mask_path(self, name: str) -> Path:
        # excel mode uses mask_<name> convention
        mask_basename = f"mask_{name}" if self.excel else name
        return self.mask_dir / f"{mask_basename}{self.mask_ext}"

    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        name    = self.name_list[index]
        img_np  = self._load_rgb(self._img_path(name))   # (H,W,3) float32 [0,255]
        mask_np = self._load_rgb(self._mask_path(name))  # (H,W,3) float32 [0,255]  <<<< RGB mask preserved

        tfm      = self.transforms_train if self.stage == "train" else self.transforms_eval
        augmented = tfm(image=img_np, mask=mask_np)
        img_t     = augmented["image"]  # (3,H,W) float32 [0,1]
        mask_t    = augmented["mask"]   # (3,H,W) float32 [0,1]

        return {"name": name, "img": img_t, "seg": mask_t}

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.name_list)