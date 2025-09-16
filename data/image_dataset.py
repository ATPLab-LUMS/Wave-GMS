# ------------------------------------------------------------------------------#
#
# File name                 : image_dataset.py
# Purpose                   : Custom PyTorch dataset class for loading and augmenting
#                             medical image datasets (BUS, BUSI, Kvasir-Instrument, HAM).
# Usage                     : See example in main training/validation scripts.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__ import annotations

import pickle
import logging
import numpy                  as np
import albumentations         as A

from PIL                      import Image
from pathlib                  import Path
from albumentations.pytorch   import ToTensorV2
from torch.utils.data         import Dataset
# ------------------------------------------------------------------------------#


# --------------------------- Dataset: Image Loader -----------------------------#
class Image_Dataset(Dataset):
    """Custom dataset for loading images and segmentation masks with Albumentations."""

    def __init__(self,
                 pickle_file_path: str | Path | None = None,
                 stage: str = "train",
                 ham: bool = False,
                 img_size: int = 224,
                 img_ext: str = ".png",
                 mask_ext: str = ".png") -> None:
        super().__init__()

        if pickle_file_path is None:
            raise ValueError("pickle_file_path must be provided")

        with open(pickle_file_path, "rb") as file:
            loaded_dict = pickle.load(file)

        pickle_dir        = Path(pickle_file_path).parent
        self.ham          = ham
        self.img_path     = pickle_dir / "images"
        self.mask_path    = pickle_dir / "masks"
        self.img_size     = img_size
        self.stage        = stage
        self.name_list    = loaded_dict[stage]["name_list"]
        self.transform    = self.get_transforms()
        logging.info("%s set num: %d", stage, len(self.name_list))

        del loaded_dict

        self.img_ext      = img_ext
        self.mask_ext     = mask_ext

    # --------------------------- Data Augmentations ----------------------------#
    def get_transforms(self):
        """Return Albumentations transforms for training/validation."""
        if self.stage == "train":
            transforms = A.Compose([
                A.ToFloat(max_value=255.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1,
                              saturation=0.1, hue=0.1, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1,
                                   rotate_limit=20, p=0.4),
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])
        else:
            transforms = A.Compose([
                A.ToFloat(max_value=255.0),
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])
        return transforms

    # --------------------------- Item Loader -----------------------------------#
    def __getitem__(self, index: int):
        """Load and return one image-mask pair after augmentation."""
        name = self.name_list[index]

        if self.ham:
            mask_name = f"{name}_segmentation" # HAM dataset specific naming
        else:
            mask_name = name

        seg_image   = Image.open(self.mask_path / f"{mask_name}{self.mask_ext}").convert("RGB")
        seg_data    = np.array(seg_image).astype(np.float32)

        img_image   = Image.open(self.img_path / f"{name}{self.img_ext}").convert("RGB")
        img_data    = np.array(img_image).astype(np.float32)

        augmented   = self.transform(image=img_data, mask=seg_data)
        aug_img     = augmented["image"]
        aug_seg     = augmented["mask"]

        if aug_seg.ndim == 2:  # add channel dimension if missing
            aug_seg = aug_seg.unsqueeze(0)

        return {"name": name, "img": aug_img, "seg": aug_seg}

    # --------------------------- Dataset Length --------------------------------#
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.name_list)