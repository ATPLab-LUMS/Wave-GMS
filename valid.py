# ------------------------------------------------------------------------------#
#
# File name                 : valod.py
# Purpose                   : Main Inference Loop for GMS
# Usage                     : python valid.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch, os, time, logging, argparse

import numpy            as np
import pandas           as pd

from tqdm               import tqdm
from PIL                import Image
from einops             import rearrange
from torch.utils.data   import DataLoader

from data               import *
from utils              import *
from networks           import *
from diffusers          import AutoencoderTiny
from configs            import config

# --------------------------- Main validation loop -----------------------------#
def run_validator():
    # ----- Get config and make relevant paths -----#
    args                 = argparse.Namespace(config = 'config.py')
    save_seg_img_path    = os.path.join(PRED_MASKS_PATH, 'binary')
    save_seg_logits_path = os.path.join(PRED_MASKS_PATH, 'logits')

    os.makedirs(SNAPSHOT_PATH, exist_ok = True)
    os.makedirs(save_seg_img_path, exist_ok = True)
    os.makedirs(save_seg_logits_path, exist_ok = True)
    os.makedirs(LOG_PATH, exist_ok = True)

    # ----- Hardware, seed & Precision -----#
    gpus = ','.join([str(i) for i in GPUS])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    seed_reproducer(SEED)
    np_dtype, torch_dtype = get_precision_dtypes(PRECISION)

    # --------- Logging setup -----#
    open_log(args, config)
    logging.info(f"Using config for Valid: {args.config}")
    CONFIG_VARS = {k: v for k, v in config.__dict__.items() if k.isupper()}
    print_options(CONFIG_VARS)

    # ---- DataLoader ----
    valid_dataset    = ImageDataset(PICKLE_FILE_PATH, stage = 'test', precision = PRECISION)
    valid_dataloader = DataLoader(valid_dataset, batch_size = 1, pin_memory = True, drop_last = False, shuffle = False)

    # ---- Load Models with required device and dtype ----
    mapping_model = get_cuda(
        ResAttnUNet_DS(**MODEL_PARAMS)
    ).to(dtype = torch_dtype)

    mapping_model = load_checkpoint(mapping_model, MODEL_WEIGHT_PATH)
    mapping_model.eval()

    # Get a glimpse of the state dict of the mapping model
    # print("Mapping Model State Dict Keys:", mapping_model.state_dict().keys())

    # Modular VAE loading (dtype, device, freeze are all config-controlled)
    vae_model = load_pretrained_model(
        model_cls               = AutoencoderTiny,
        pretrained_name_or_path = "madebyollin/taesd",
        dtype                   = torch_dtype,
        device                  = "cuda",
        freeze                  = True,
    )
    vae_model.eval()
    scale_factor = VAE_SCALE_FACTOR

    # ---- Validation loop ----
    mse_loss    = torch.nn.MSELoss(reduction='mean')
    epoch_start = time.time()
    name_list, T_loss_valid = [], []

    for batch_data in tqdm(valid_dataloader, desc = 'Valid: '):
        img_rgb = batch_data['img']
        img_rgb = img_rgb / 255.0 # [ADDED] V.V.V Imp!  --> SCALE CORRECTION
        img_rgb = 2. * img_rgb - 1.
        seg_raw = batch_data['seg'].permute(0, 3, 1, 2) / 255.0
        seg_rgb = 2. * seg_raw - 1.
        name    = batch_data['name'][0]
        name_list.append(name)

        with torch.no_grad():
            img_latent_mean      = vae_model.encode(get_cuda(img_rgb)).latents
            seg_latent_mean      = vae_model.encode(get_cuda(seg_rgb)).latents
            out_latent_mean_dict = mapping_model(img_latent_mean)

            loss_Rec = W_REC * mse_loss(out_latent_mean_dict['out'], seg_latent_mean)
            pred_seg = vae_decode(vae_model, out_latent_mean_dict['out'], scale_factor)
            pred_seg = pred_seg.repeat(1, 3, 1, 1)

            x_logits = rearrange(pred_seg.squeeze().cpu().numpy(), 'c h w -> h w c')
            x_binary = np.where(x_logits > 0.5, 1, 0) * 255.
            x_binary = Image.fromarray(x_binary.astype(np.uint8))

            # Save both logits and binary
            save_binary_and_logits(x_logits, x_binary, name, save_seg_img_path, save_seg_logits_path)

            T_loss_valid.append(loss_Rec.item())

    T_loss_valid = np.mean(T_loss_valid)
    logging.info("Valid:\nloss: {:.4f}".format(T_loss_valid))

    # ---- Metrics calculation ----
    csv_path  = os.path.join(SNAPSHOT_PATH, 'results.csv')
    true_path = os.path.join(os.path.dirname(PICKLE_FILE_PATH), 'masks')

    pred_binary_path = save_seg_img_path
    pred_logits_path = save_seg_logits_path

    name_list = sorted(os.listdir(save_seg_img_path))
    # Remove IMG_FORMAT from names
    name_list = [x.replace(IMG_FORMAT, '') for x in name_list]
    name_list = [x.replace('_binary', '') for x in name_list]

    dsc_list, iou_list, hd95_list = [], [], []
    ssim_list, ssim_region_list, ssim_object_list, ssim_combined_list = [], [], [], []

    for case_name in tqdm(name_list):
        seg_binary   = load_img(os.path.join(pred_binary_path, case_name + '_binary' +  IMG_FORMAT), img_size = IMG_SIZE, dtype_resize = np_dtype)
        seg_logits   = load_img(os.path.join(pred_logits_path, case_name + '_logits' + IMG_FORMAT), img_size = IMG_SIZE, dtype_resize = np_dtype)
        seg_true     = load_img(os.path.join(true_path, case_name + IMG_FORMAT), img_size = IMG_SIZE, dtype_resize = np_dtype)

        # Calculate all metrics
        results = all_metrics(seg_binary, seg_logits, seg_true)

        # Append all scores into respective list by inexing the results dict
        dsc_list.append(results['DSC'])
        iou_list.append(results['IoU'])
        hd95_list.append(results['HD95'])

        ssim_list.append(results['SSIM'])
        ssim_region_list.append(results['SSIM_region'])
        ssim_object_list.append(results['SSIM_object'])
        ssim_combined_list.append(results['SSIM_combined'])

    # Add mean/std
    name_list.extend(['Avg', 'Std'])

    dsc_list.extend([np.mean(dsc_list), np.std(dsc_list, ddof=1)])
    iou_list.extend([np.mean(iou_list), np.std(iou_list, ddof=1)])
    hd95_list.extend([np.mean(hd95_list), np.std(hd95_list, ddof=1)])

    ssim_list.extend([np.mean(ssim_list), np.std(ssim_list, ddof=1)])
    ssim_region_list.extend([np.mean(ssim_region_list), np.std(ssim_region_list, ddof=1)])
    ssim_object_list.extend([np.mean(ssim_object_list), np.std(ssim_object_list, ddof=1)])
    ssim_combined_list.extend([np.mean(ssim_combined_list), np.std(ssim_combined_list, ddof=1)])

    df = pd.DataFrame({'Name': name_list, 'DSC': dsc_list, 'IoU': iou_list, 'HD95': hd95_list, 'SSIM': ssim_list,
                       'SSIM_region': ssim_region_list, 'SSIM_object': ssim_object_list,
                       'SSIM_combined': ssim_combined_list})
    df.to_csv(csv_path, index=False)

    logging.info("DSC: {:.4f}, IOU: {:.4f}, HD95: {:.2f}".format(dsc_list[-2], iou_list[-2], hd95_list[-2]))
    logging.info('Time Taken: %d sec' % (time.time() - epoch_start))
    logging.info('\n')

# -----------------------------------------------------------------------#
if __name__ == "__main__":
    run_validator()

# -------------------------------- End ----------------------------------#

# Basic Package
# import torch
# import argparse
# import numpy as np
# import yaml
# import logging
# import time
# import os
# import pandas as pd
# from einops import rearrange
# from omegaconf import OmegaConf
# from tqdm import tqdm
# from PIL import Image
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix
# from configs import config
# # Own Package
# from diffusers import AutoencoderTiny
# from data import *
# from utils import *
# from networks import *

# def run_validator():
#     # ----- Get config and make relevant paths -----#
#     args                 = argparse.Namespace(config = 'config.py')
#     save_seg_img_path    = os.path.join(PRED_MASKS_PATH, 'binary')
#     save_seg_logits_path = os.path.join(PRED_MASKS_PATH, 'logits')

#     os.makedirs(SNAPSHOT_PATH, exist_ok = True)
#     os.makedirs(save_seg_img_path, exist_ok = True)
#     os.makedirs(save_seg_logits_path, exist_ok = True)
#     os.makedirs(LOG_PATH, exist_ok = True)

#     # ----- Hardware, seed & Precision -----#
#     gpus = ','.join([str(i) for i in GPUS])
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpus
#     seed_reproducer(SEED)
#     np_dtype, torch_dtype = get_precision_dtypes(PRECISION)

#     # --------- Logging setup -----#
#     open_log(args, config)
#     logging.info(f"Using config for Valid: {args.config}")
#     CONFIG_VARS = {k: v for k, v in config.__dict__.items() if k.isupper()}
#     print_options(CONFIG_VARS)

#     # ---- DataLoader ----
#     valid_dataset    = ImageDataset(PICKLE_FILE_PATH, stage = 'test', precision = PRECISION)
#     valid_dataloader = DataLoader(valid_dataset, batch_size = 1, pin_memory = True, drop_last = False, shuffle = False)

#     # ---- Load Models with required device and dtype ----
#     mapping_model = get_cuda(
#         ResAttnUNet_DS(**MODEL_PARAMS)
#     ).to(dtype = torch_dtype)

#     mapping_model = load_checkpoint(mapping_model, MODEL_WEIGHT_PATH)
#     mapping_model.eval()

#     # Modular VAE loading (dtype, device, freeze are all config-controlled)
#     vae_model = load_pretrained_model(
#         model_cls               = AutoencoderTiny,
#         pretrained_name_or_path = "madebyollin/taesd",
#         dtype                   = torch_dtype,
#         device                  = "cuda",
#         freeze                  = True,
#     )
#     vae_model.eval()
#     scale_factor = VAE_SCALE_FACTOR

#     # ---- Validation loop ----
#     mse_loss    = torch.nn.MSELoss(reduction='mean')
#     epoch_start = time.time()
#     name_list, T_loss_valid = [], []

#     ### Validation phase
#     for batch_data in tqdm(valid_dataloader, desc='Valid: '):
#         img_rgb = batch_data['img']
#         img_rgb = img_rgb / 255.0 # [CHANGED] V.V.V Imp!  --> SCALE CORRECTION
#         img_rgb = 2. * img_rgb - 1.
#         seg_raw = batch_data['seg']
#         seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
#         seg_rgb = 2. * seg_raw - 1.
#         name = batch_data['name'][0]
#         name_list.append(name)

#         with torch.no_grad():
#             img_latent_mean, seg_latent_mean = vae_model.encode(get_cuda(img_rgb)).latents, vae_model.encode(get_cuda(seg_rgb)).latents
#             out_latent_mean_dict = mapping_model(img_latent_mean)

#             loss_Rec = W_REC * mse_loss(out_latent_mean_dict['out'], seg_latent_mean)

#             pred_seg = vae_decode(vae_model, out_latent_mean_dict['out'], scale_factor)
#             pred_seg = pred_seg.repeat(1, 3, 1, 1) # [CHANGED] --> repeat to match RGB channels

#             x_sample = rearrange(pred_seg.squeeze().cpu().numpy(), 'c h w -> h w c')
#             x_sample = np.where(x_sample > 0.5, 1, 0)
#             x_sample = 255. * x_sample
#             img = Image.fromarray(x_sample.astype(np.uint8))
#             # [CHANGED] --> Question: Is the segmentation being saved in grayscale or RGB? Also changed png to jpg
#             img.save(os.path.join(save_seg_img_path, name + '.png'))

#             T_loss_valid.append(loss_Rec.item())

#     T_loss_valid = np.mean(T_loss_valid)

#     logging.info("Valid:")
#     logging.info("loss: {:.4f}".format(T_loss_valid))

#     ### load masks & compute dsc and iou
#     csv_path  = os.path.join(SNAPSHOT_PATH, 'results.csv')
#     pred_path = save_seg_img_path
#     true_path = os.path.join(os.path.dirname(PICKLE_FILE_PATH), 'masks')

#     name_list = sorted(os.listdir(pred_path))
#     name_list = [x.replace('.png', '') for x in name_list] # [CHANGED] --> changed from .png to .png
#     name_list = [x.replace('_segmentation', '') for x in name_list]

#     dsc_list = []
#     iou_list = []

#     for case_name in tqdm(name_list):
#         # [CHANGED] --> changed from .png to .png
#         seg_binary = load_img(os.path.join(pred_path, case_name + '.png'))
#         seg_true = load_img(os.path.join(true_path, case_name + '.png'))

#         preds = np.array(seg_binary).reshape(-1)
#         gts = np.array(seg_true).reshape(-1)

#         y_pre = np.where(preds>=0.5, 1, 0)
#         y_true = np.where(gts>=0.5, 1, 0)

#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

#         dsc_list.append(f1_or_dsc)
#         iou_list.append(miou)

#     # MEAN & Std Value
#     name_list.extend(['Avg', 'Std'])
#     dsc_list.extend([np.mean(dsc_list), np.std(dsc_list, ddof=1)])
#     iou_list.extend([np.mean(iou_list), np.std(iou_list, ddof=1)])

#     df = pd.DataFrame({
#         'Name': name_list,
#         'DSC':  dsc_list,
#         'IoU': iou_list
#     })
#     df.to_csv(csv_path, index=False)

#     logging.info("DSC: {:.4f}, IOU: {:.4f}".format(dsc_list[-2], iou_list[-2]))
#     logging.info('Time Taken: %d sec' % (time.time() - epoch_start))
#     logging.info('\n')

# if __name__ == '__main__':
#     run_validator()
