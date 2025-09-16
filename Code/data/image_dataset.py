# ------------------------------------------------------------------------------#
#
# File name                 : image_dataset.py
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

import os, pickle, logging, random

import numpy          as np
import pandas         as pd
import albumentations as A

from __future__             import annotations
from PIL                    import Image
from albumentations.pytorch import ToTensorV2
from pathlib                import Path
from typing                 import Dict, List
from torch.utils.data       import Dataset

# train and test split (90 - 10)

# --------------------------- Split helpers ----------------------------------#

def _excel_splits_exist(root: Path) -> bool:
    return all((root / fname).is_file() for fname in ("Train_ID.xlsx", "Val_ID.xlsx", "Test_ID.xlsx"))


def _build_split_from_excels(root: Path) -> Dict[str, Dict[str, List[str]]]:
    """Parse Train/Val/Test Excel sheets and build split dict.

    Returns
    -------
    dict
        ``{"train": {"name_list": [...]}, "val": {...}, "test": {...}}``
    """
    split_keys   = ["train", "val", "test"]
    excel_files  = ["Train_ID.xlsx", "Val_ID.xlsx", "Test_ID.xlsx"]
    split_dict   = {}

    for key, fname in zip(split_keys, excel_files):
        df    = pd.read_excel(root / fname)
        masks = df["Image"].astype(str).tolist()
        # Strip prefix "mask_" and extension → base name
        bases = sorted({os.path.splitext(m)[0].replace("mask_", "") for m in masks})
        split_dict[key] = {"name_list": bases}
        logging.info(f"Loaded {len(bases):5d} items for {key:>5s} from {fname}.")

    return split_dict

def generate_pickle_excel(root_dir: str | Path, pickle_name: str = "QaTar_19_train_val_test_names.pkl") -> Path:
    """Create (or return existing) pickle file with train/val/test splits."""
    root_dir = Path(root_dir)
    pkl_path = root_dir / pickle_name

    # Return early if already present
    if pkl_path.is_file():
        logging.info(f"Split pickle already exists → {pkl_path}")
        return pkl_path

    # Decide strategy
    if _excel_splits_exist(root_dir):
        split_dict = _build_split_from_excels(root_dir)

    # Save
    with open(pkl_path, "wb") as f:
        pickle.dump(split_dict, f)
    logging.info(f"Saved split pickle → {pkl_path}")

    return pkl_path

def generate_pickle_default(root_dir: str | Path, split: float = 0.9, name: str = 'kvasir-seg_train_test_names.pkl') -> Path:
    root_dir = Path(root_dir)
    # Get img and mask files
    img_dir  = root_dir / 'images'
    mask_dir = root_dir / 'masks'

    # Get list of image and mask files
    img_files  = [p.name for p in img_dir.glob('*.png')]
    mask_files = [p.name for p in mask_dir.glob('*.png')]

    # Get a random batch of indices of img_dir (for train) and its corresponding mask_file indidces and then the
    # remaining goes to test.

    # In the end a .pkl object of a dict should be saved in the root_dir which should contain two keys 'train' and 'test'
    # where each key a dict value. The dict value should have key 'name_list' and its value should be a list of the file names
    # chosen for the train/test

    # E.g
    # {'train': {'name_list': [file_name1_train.jpg, file_name2_train.jpg]}, 'test': {'name_list': [file_name1_test.jpg, file_name2_test.jpg]}}

    # Extract base filenames (without extension) to match pairs
    img_bases  = [Path(f).stem for f in img_files]
    mask_bases = [Path(f).stem for f in mask_files]

    # Find common base filenames (ensure paired image-mask)
    common_bases = sorted(list(set(img_bases) & set(mask_bases)))

    # Shuffle for random split
    random.seed(42)  # For reproducibility
    random.shuffle(common_bases)

    # Split into train and test (90-10)
    train_size  = int(split * len(common_bases))
    train_bases = common_bases[:train_size]
    test_bases  = common_bases[train_size:]

    # Create dictionary for pickle file
    pickle_dict = {
        'train': {'name_list': train_bases},
        'test': {'name_list': test_bases}
    }

    # Save pickle file in root_dir
    pickle_path = root_dir / name
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_dict, f)

    logging.info(f"Pickle file saved at {pickle_path}")
    logging.info(f"Train set: {len(train_bases)} files, Test set: {len(test_bases)} files")

    return pickle_path

class Image_Dataset(Dataset):
    def __init__(self,
                 pickle_file_path: str | Path | None = None,
                 root_dir: str | Path = 'QaTar-19',
                 stage: str = 'train',
                 excel: bool = False,
                 ham: bool = False,
                 img_size: int = 224,
                 img_ext: str = '.png',
                 mask_ext: str = '.png'
                 ) -> None:
        super().__init__()

        if pickle_file_path is None:
           root_dir = Path('Dataset') / root_dir
           if excel:
               pickle_file_path = generate_pickle_excel(root_dir)
           else:
               pickle_file_path = generate_pickle_default(root_dir)

        with open(pickle_file_path, 'rb') as file:
            loaded_dict = pickle.load(file)

        pickle_dir = Path(pickle_file_path).parent
        self.excel             = excel
        self.ham               = ham
        self.img_path          = pickle_dir / 'images'
        self.mask_path         = pickle_dir / 'masks'
        self.img_size          = img_size
        self.stage             = stage
        self.name_list         = loaded_dict[stage]['name_list']
        self.transform         = self.get_transforms()
        logging.info('{} set num: {}'.format(stage, len(self.name_list)))

        del loaded_dict

        self.img_ext = img_ext
        self.mask_ext = mask_ext

    
    def get_transforms(self):
        if self.stage == 'train':
            transforms = A.Compose([
                A.ToFloat(max_value=255.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=20, p=0.4),

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
        # f
    def __getitem__(self, index):
        name = self.name_list[index]
        # load img & seg
        
        # mask are being loaded as RGB
        if self.excel:
            mask_name = f"mask_{name}"
        elif self.ham:
            mask_name = f'{name}_segmentation'
        else:
            mask_name = name
        seg_image = Image.open(self.mask_path / f"{mask_name}{self.mask_ext}").convert("RGB") # Load as Grayscale
        seg_data  = np.array(seg_image).astype(np.float32)
        img_image = Image.open(self.img_path / f"{name}{self.img_ext}").convert("RGB")
        img_data  = np.array(img_image).astype(np.float32)

        augmented = self.transform(image=img_data, mask=seg_data)

        aug_img = augmented['image']

        aug_seg = augmented['mask']

        # Add channel dimension to mask if it's missing
        if aug_seg.ndim == 2:
            aug_seg = aug_seg.unsqueeze(0)
        return {
            'name': name,
            'img': aug_img,
            'seg': aug_seg
        }

    def __len__(self):
        return len(self.name_list)