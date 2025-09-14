# ------------------------------------------------------------------------------#
#
# File name      : config.py
# Purpose        : Centralized config as Python variables, supports dynamic paths
# Usage          : Import directly
#
# Authors        : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email          : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                  hassan.mohyuddin@lums.edu.pk
#
# Last Modified  : June 23, 2025

# --------------------------- Experiment setup ---------------------------------#
DATASET           = "busi"                          # "bus", "kvasir-instrument", etc.
PHASE             = "train"                         # "train" or "valid"
PRECISION         = "float32"                       # "float32", "float16", "bfloat16"
IMG_SIZE          = 224                             # Dimension for resized images, e.g., 224 for 224x224
SEED              = 2333
VAE_SCALE_FACTOR  = 1.0
VAE_MODE          = {'eval': True, 'train': False}  # 'train' or 'eval' mode for VAE
IMG_FORMAT        = ".png"                          # 'jpg', 'png', 'tif', etc.

# --------------------------- Base paths ---------------------------------------#
BASE_DATA_DIR     = "./Dataset"
BASE_CKPT_DIR     = "./ckpt"

# --------------------------- Derived paths ------------------------------------#
# f'/media/ee/DATA/Talha_Nehal/Diffusion-Codes/GMS/ckpt/tiny_vae_busi/epochs_200/checkpoints/{CHOSEN_STRATEGY}.pth'
BRIEF_DESCRIPTION = 'test_code_2'
CHOSEN_STRATEGY   = 'best_valid_dice'  # 'latent_diffusion', 'diffusion', etc.
SNAPSHOT_PATH     = f"{BASE_CKPT_DIR}/{DATASET}/{BRIEF_DESCRIPTION}"
LOG_PATH          = f"{SNAPSHOT_PATH}/logs"
PICKLE_FILE_PATH  = f"{BASE_DATA_DIR}/{DATASET}/{DATASET}_train_test_names.pkl"
MODEL_WEIGHT_PATH = f'/media/ee/DATA/Talha_Nehal/Diffusion-Codes/GMS/ckpt/tiny_vae_busi/epochs_200/checkpoints/{CHOSEN_STRATEGY}.pth' #f"{SNAPSHOT_PATH}/checkpoints/{CHOSEN_STRATEGY}.pth"
PRED_MASKS_PATH   = f"{SNAPSHOT_PATH}/predicted_masks_{CHOSEN_STRATEGY}"

# --------------------------- Training settings --------------------------------#
EPOCHS            = 200
BATCH_SIZE        = 16
SAVE_FREQ         = 25
LR                = 0.002
NUM_WORKERS       = 8
GPUS              = [0]

# --------------------------- Validation settings ------------------------------#
VAL_BATCH_SIZE    = 1
VAL_NUM_WORKERS   = 4

# --------------------------- Model settings -----------------------------------#
MODEL_PARAMS = {
    "in_channel"          : 4,
    "out_channels"        : 4,
    "num_res_blocks"      : 2,
    "ch"                  : 32,
    "ch_mult"             : [1, 2, 4, 4],
}


# --------------------------- Loss weights -------------------------------------#
W_REC             = 1
W_DICE            = 1

# --------------------------------- End ----------------------------------------#
