# ------------------------------------------------------------------------------#
#
# File name                 : train.py
# Purpose                   : Main training loop for GMS
# Usage                     : python train.py --config configs/experiment.yaml
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module imports ----------------------------------#
import os, time, yaml, torch, logging, argparse

import numpy            as np

from tqdm               import tqdm
from torch.utils.data   import DataLoader
from monai.losses.dice  import DiceLoss

from data               import *
from utils              import *
from configs            import config
from networks           import *

from diffusers          import AutoencoderTiny # (mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers)
from tensorboardX       import SummaryWriter


# --------------------- Main training function ---------------------------------#
def run_trainer() -> None:
    """
    Complete training loop for the latent diffusion segmentation pipeline.
    Loads config, initializes logger, datasets, models, optimizers, scheduler,
    runs training/validation, and handles checkpointing.
    """
    # ------------- Parse args & flatten config ---------------------------------------#
    args = argparse.Namespace(config = 'config.py')
    # Dynamically patch snapshot and log paths with timestamp
    os.makedirs(SNAPSHOT_PATH, exist_ok = True)
    os.makedirs(LOG_PATH, exist_ok = True)

    # ------------- Hardware, seed & precision -------------------------------------------#
    gpus = ",".join([str(i) for i in GPUS])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    seed_reproducer(SEED)
    np_dtype, torch_dtype = get_precision_dtypes(PRECISION)

    # ------------- Logging setup ---------------------------------------------#
    open_log(args, config)
    logging.info(f"Using config for Train: {args.config}")
    CONFIG_VARS = {k: v for k, v in config.__dict__.items() if k.isupper()}
    print_options(CONFIG_VARS)

    # ------------- TensorBoard setup -----------------------------------------#
    writer  = SummaryWriter(LOG_PATH)
    ds_list = ["level2", "level1", "out"]  # Multi-scale levels

    # ------------- Datasets/Dataloaders -------------------------------------#
    train_dataset = ImageDataset(PICKLE_FILE_PATH, stage="train", precision=PRECISION)
    valid_dataset = ImageDataset(PICKLE_FILE_PATH, stage="test", precision=PRECISION)

    train_loader = DataLoader(
        train_dataset,
        batch_size            =  BATCH_SIZE,
        pin_memory            =  True,
        drop_last             =  True,
        shuffle               =  True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size            = BATCH_SIZE,
        pin_memory            = True,
        drop_last             = False,
        shuffle               = False,
    )

    # ------------- Model definitions ----------------------------------------#
    mapping_model = get_cuda(ResAttnUNet_DS(**MODEL_PARAMS)).to(dtype=torch_dtype)

    # Modular VAE loading (dtype, device, freeze are all config-controlled)
    vae_model = load_pretrained_model(
        model_cls               =   AutoencoderTiny,
        pretrained_name_or_path =   "madebyollin/taesd",
        dtype                   =   torch_dtype,
        device                  =   "cuda",
        freeze                  =   True,
    )
    scale_factor = VAE_SCALE_FACTOR

    # ------------- Optimizer & scheduler ------------------------------------#
    optimizer = torch.optim.AdamW(mapping_model.parameters(), lr=LR)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs = 5, max_epochs = EPOCHS
    )

    # ------------- Loss functions -------------------------------------------#
    mse_loss  = torch.nn.MSELoss(reduction = "mean")
    dice_loss = DiceLoss()

    # ------------- Training/Validation state --------------------------------#
    iter_num                = 0
    best_valid_loss         = np.inf
    best_valid_loss_rec     = np.inf
    best_valid_dice         = 0
    best_valid_dice_epoch   = 0
    best_valid_loss_dice    = np.inf

    ramup_length = 15 # for the weight for loss dice
    w2           = W_DICE  # inital value for weight of loss dice

    # =========================================================================
    #                               TRAINING LOOP
    # =========================================================================
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        mapping_model.train()
        vae_model.eval() if VAE_MODE["eval"] else vae_model.train()
        T_loss, T_loss_Rec, T_loss_Dice = [], [], []
        T_loss_valid, T_loss_Rec_valid, T_loss_Dice_valid, T_Dice_valid = [], [], [], []

        # =================== Training phase =================== #
        for batch_data in tqdm(train_loader, desc=f"Train (epoch {epoch})"):
            img_rgb = 2.0 * batch_data["img"] - 1.0
            img_rgb = img_rgb / 255.0 # [ADDED] V.V.V Imp!  --> SCALE CORRECTION
            seg_raw = batch_data["seg"].permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2.0 * seg_raw - 1.0
            seg_img = torch.mean(seg_raw, dim = 1, keepdim = True)
            name    = batch_data["name"]

            img_latent_mean_aug = vae_model.encode(get_cuda(img_rgb)).latents
            seg_latent_mean     = vae_model.encode(get_cuda(seg_rgb)).latents

            out_latent_mean_dict = mapping_model(img_latent_mean_aug)

            # w2 = gaussian_rampup(epoch, ramup_length)
            # print(f'w2 value: {w2}')

            loss_Rec             = W_REC * get_multi_loss( # instead of W_REC
                mse_loss,
                out_latent_mean_dict,
                seg_latent_mean,
                is_ds     = True,
                key_list  = ds_list,
            )

            pred_seg_dict = {
                level: vae_decode(vae_model, out_latent_mean_dict[level], scale_factor)
                for level in ds_list
            }

            loss_Dice = W_DICE * get_multi_loss(  # instead of W_DICE
                dice_loss,
                pred_seg_dict,
                get_cuda(seg_img),
                is_ds        = True,
                key_list     = ds_list,
            )

            loss = loss_Rec + loss_Dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num     += 1
            if iter_num % 10 == 0:
                writer.add_scalar("loss/loss", loss, iter_num)
                writer.add_scalar("loss/loss_Rec", loss_Rec, iter_num)
                writer.add_scalar("loss/loss_Dice", loss_Dice, iter_num)

            T_loss.append(loss.item())
            T_loss_Rec.append(loss_Rec.item())
            T_loss_Dice.append(loss_Dice.item())

        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # ---- Logging training ----
        T_loss      = np.mean(T_loss)
        T_loss_Rec  = np.mean(T_loss_Rec)
        T_loss_Dice = np.mean(T_loss_Dice)
        logging.info(
            f"Train: loss: {T_loss:.4f}, loss_Rec: {T_loss_Rec:.4f}, loss_Dice: {T_loss_Dice:.4f}"
        )
        writer.add_scalar("train/loss", T_loss, epoch)
        writer.add_scalar("train/loss_Rec", T_loss_Rec, epoch)
        writer.add_scalar("train/loss_Dice", T_loss_Dice, epoch)

        # =================== Validation phase =================== #
        mapping_model.eval()
        vae_model.eval()
        for batch_data in tqdm(valid_loader, desc="Valid: "):
            img_rgb = 2.0 * batch_data["img"] - 1.0
            img_rgb = img_rgb / 255.0 # [ADDED] V.V.V Imp!  --> SCALE CORRECTION
            seg_raw = batch_data["seg"].permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2.0 * seg_raw - 1.0
            seg_img = torch.mean(seg_raw, dim=1, keepdim=True)
            name = batch_data["name"]

            with torch.no_grad():
                img_latent_mean = vae_model.encode(get_cuda(img_rgb)).latents
                seg_latent_mean = vae_model.encode(get_cuda(seg_rgb)).latents

                out_latent_mean_dict = mapping_model(img_latent_mean)
                pred_seg = vae_decode(
                    vae_model, out_latent_mean_dict["out"], scale_factor
                )

                loss_Rec = W_REC * mse_loss(
                    out_latent_mean_dict["out"], seg_latent_mean
                )
                loss_Dice = w2 * dice_loss(pred_seg, get_cuda(seg_img))

                loss = loss_Rec + loss_Dice

                # ---- Dice calculation ----
                pred_seg     = pred_seg.cpu()
                reduce_axis  = list(range(1, len(seg_img.shape)))
                intersection = torch.sum(seg_img * pred_seg, dim=reduce_axis)
                y_o          = torch.sum(seg_img, dim=reduce_axis)
                y_pred_o     = torch.sum(pred_seg, dim=reduce_axis)
                denominator  = y_o + y_pred_o
                dice_raw     = (2.0 * intersection) / denominator
                dice_value   = dice_raw.mean()

                T_Dice_valid.append(dice_value.item())
                T_loss_valid.append(loss.item())
                T_loss_Rec_valid.append(loss_Rec.item())
                T_loss_Dice_valid.append(loss_Dice.item())

        # ---- Logging validation ----
        T_Dice_valid        = np.mean(T_Dice_valid)
        T_loss_valid        = np.mean(T_loss_valid)
        T_loss_Rec_valid    = np.mean(T_loss_Rec_valid)
        T_loss_Dice_valid   = np.mean(T_loss_Dice_valid)

        writer.add_scalar("valid/dice", T_Dice_valid, epoch)
        writer.add_scalar("valid/loss", T_loss_valid, epoch)
        writer.add_scalar("valid/loss_Rec", T_loss_Rec_valid, epoch)
        writer.add_scalar("valid/loss_Dice", T_loss_Dice_valid, epoch)

        logging.info(
            f"Valid: loss: {T_loss_valid:.4f}, loss_Rec: {T_loss_Rec_valid:.4f}, "
            f"loss_Dice: {T_loss_Dice_valid:.4f}, Dice: {T_Dice_valid:.4f}"
        )

        # ---- Model checkpointing ----
        if T_Dice_valid > best_valid_dice:
            save_checkpoint(mapping_model, "best_valid_dice.pth", SNAPSHOT_PATH)
            best_valid_dice         = T_Dice_valid
            best_valid_dice_epoch   = epoch
            logging.info("Save best valid Dice !")

        if T_loss_valid < best_valid_loss:
            save_checkpoint(mapping_model, "best_valid_loss.pth", SNAPSHOT_PATH)
            best_valid_loss = T_loss_valid
            logging.info("Save best valid Loss All !")

        if T_loss_Rec_valid < best_valid_loss_rec:
            save_checkpoint(mapping_model, "best_valid_loss_rec.pth", SNAPSHOT_PATH)
            best_valid_loss_rec = T_loss_Rec_valid
            logging.info("Save best valid Loss Rec !")

        if T_loss_Dice_valid < best_valid_loss_dice:
            save_checkpoint(mapping_model, "best_valid_loss_dice.pth", SNAPSHOT_PATH)
            best_valid_loss_dice = T_loss_Dice_valid
            logging.info("Save best valid Loss Dice !")

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(
                mapping_model,
                f"latent_mapping_model_epoch_{epoch:04d}.pth",
                SNAPSHOT_PATH,
            )

        logging.info("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
        logging.info(
            f"epoch {epoch} / {EPOCHS} \t Time Taken: {int(time.time() - epoch_start_time)} sec"
        )
        logging.info(
            f"best valid dice: {best_valid_dice:.4f} at epoch: {best_valid_dice_epoch}"
        )
        logging.info("\n")
    writer.close()


# -----------------------------------------------------------------------#
if __name__ == "__main__":
    run_trainer()

# -------------------------------- End ----------------------------------#
