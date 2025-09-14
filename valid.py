# Basic Package
import torch
import argparse
import numpy as np
import yaml
import logging
import time
import os
import pandas as pd
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from medpy import metric
# Own Package
from data.image_dataset import Image_Dataset
from tools.utils import *
from tools.load_ckpt import *
from tools.get_logger import open_log
from tools.metrics import all_metrics
from tools.load_ckpt import get_tiny_autoencoder, get_lite_vae
from modules.latent_mapping_model import ResAttnUNet_DS
from modules.models.distributions import DiagonalGaussianDistribution
from modules.sft_lmm import *
from modules.guidance import *

def save_binary_and_logits(x_logits, x_binary, name, save_seg_img_path, save_seg_logits_path, IMG_FORMAT = '.png'):
    """ Saves binary and logits images to specified path."""

    x_binary.save(os.path.join(save_seg_img_path, name + '_binary' + IMG_FORMAT))
    # Save x_logits as .png
    x_logits = (x_logits * 255).astype(np.uint8)
    x_logits = Image.fromarray(x_logits)
    x_logits.save(os.path.join(save_seg_logits_path, name + '_logits' + IMG_FORMAT))

def load_img(path, img_size = 224, dtype_resize = np.float32):
    """Loads and normalizes a grayscale mask image to [0,1], resizes to (img_size, img_size)."""

    image = Image.open(path).convert("L").resize((img_size, img_size), resample=Image.NEAREST)
    image = np.array(image).astype(dtype_resize) / 255.0
    return image

def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * mean, logvar

def vae_decode(vae_model, pred_mean, scale_factor, sd_vae = False):
    z = 1.0 / scale_factor * pred_mean
    pred_seg = vae_model.decode(z) if sd_vae else vae_model.decode(z).sample # [CHANGED] --> has channels = 3 according to config
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True) # [CHANGED] --> Taking mean across channels dimension resulting in 1 channel
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0)  # (B, 1, H, W) # [CHANGED] --> Bringing the range to (0, 1) as per Kvasir-SEG dataset
    return pred_seg

def arg_parse() -> argparse.ArgumentParser.parse_args :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/busi_valid.yaml', # [CHANGED] --> added kvasir-seg yaml
                        type=str, help='load the config file')
    args = parser.parse_args()
    return args

def run_validator() -> None:
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)

    save_seg_img_path    = os.path.join(configs['save_seg_img_path'], 'binary')
    save_seg_logits_path = os.path.join(configs['save_seg_img_path'], 'logits')

    configs['log_path'] = os.path.join(configs['snapshot_path'], 'logs')

    # Output folder and save fig folder
    os.makedirs(configs['snapshot_path'], exist_ok=True)
    os.makedirs(save_seg_img_path, exist_ok = True)
    os.makedirs(save_seg_logits_path, exist_ok = True)
    os.makedirs(configs['log_path'], exist_ok=True)

    device = 'mps' # 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set GPU ID
    if torch.cuda.is_available():
        gpus = ','.join([str(i) for i in configs['GPUs']])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Fix seed (for repeatability)
    seed_reproducer(configs['seed'])

    # Open log file
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # Get data loader
    valid_dataset    = Image_Dataset(configs['pickle_file_path'], stage='test', excel = False) # include whole dataset for generalizability experiment
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    # Define modules
    skff_module = None
    if configs['guidance_method']:
        guidance_channels_dict = {'edge': 3, 'wavelet': 3, 'dino': 384}

        mapping_model = SFT_UNet_DS(in_channels       = configs['in_channel'],
                                    out_channels      = configs['out_channels'],
                                    guidance_channels = guidance_channels_dict[configs['guidance_method']]).to(device)

        if configs['guidance_method'] == 'wavelet':
            skff_module = SKFF().to(device)
            skff_module.eval()

    else:
        mapping_model = ResAttnUNet_DS(
            in_channel=configs["in_channel"],
            out_channels=configs["out_channels"],
            num_res_blocks=configs["num_res_blocks"],
            ch=configs["ch"],
            ch_mult=configs["ch_mult"],
            ).to(device)

    mapping_model.eval()

    vae_train = True
    vae_model = None
    tiny_vae  = None

    scale_factor = 1.0

    if configs['vae_model'] == 'tiny_vae':
        logging.info("Initializing TinyVAE")
        vae_model = get_tiny_autoencoder(train = False, residual_autoencoding = False)
    elif 'litevae-s' in configs['vae_model']:
        logging.info("Initializing LiteVAE")
        tiny_vae  = get_tiny_autoencoder(train = False, residual_autoencoding = False) # for the segmentation latent and decoding at the end.
        vae_model = get_lite_vae(model_version = configs['vae_model'], train = False, freeze = True)
    elif 'sd-vae' in configs['vae_model']:
        logging.info("Initializing Stable Diffusion VAE")
        vae_model, scale_factor = get_sd_vae(train = False, freeze = True)

    if vae_train and skff_module is None:
        mapping_model, vae_model, _ = load_checkpoint(mapping_model, configs['model_weight'],
                                                   vae_model = vae_model, vae_model_load = vae_train)
    elif vae_train and skff_module is not None:
        mapping_model, vae_model, skff_module = load_checkpoint(mapping_model, configs['model_weight'],
                                                                vae_model = vae_model, vae_model_load = vae_train,
                                                                skff_model = skff_module, skff_model_load = True)
    elif not vae_train and skff_module is not None:
        mapping_model, vae_model, skff_module = load_checkpoint(mapping_model, configs['model_weight'],
                                                                vae_model = vae_model, vae_model_load = False,
                                                                skff_model = skff_module, skff_model_load = True)
    else:
        mapping_model, _, _ = load_checkpoint(mapping_model, configs['model_weight'])

    mapping_model.eval()
    vae_model.eval()
    tiny_vae.eval() if configs['vae_model'] != 'tiny_vae' and tiny_vae is not None else None
    # Getting tiny-vae (with residual_autoencoding) default: frozen and eval

    # Define loss functions
    mse_loss = torch.nn.MSELoss(reduction='mean')

    epoch_start_time = time.time()

    name_list = []

    T_loss_valid = []

    ### Validation phase
    for batch_data in tqdm(valid_dataloader, desc='Valid: '):
        img_rgb = batch_data['img'].to(device)
        img_rgb = img_rgb / 255.0 # [CHANGED] V.V.V Imp!  --> SCALE CORRECTION

        if configs['vae_model'] == 'tiny_vae' or 'sd-vae' in configs['vae_model']:
            img_rgb = 2. * img_rgb - 1.

        seg_raw = batch_data['seg'].to(device)
        seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
        seg_rgb = 2. * seg_raw - 1.

        seg_img = torch.mean(seg_rgb, dim=1, keepdim=True)  # (B, 1, H, W)

        name = batch_data['name'][0]
        name_list.append(name)

        with torch.no_grad():
            if configs['vae_model'] == 'tiny_vae':
                img_latent_mean, seg_latent_mean = (
                    vae_model.encode(img_rgb).latents,
                    vae_model.encode(seg_rgb).latents,
                )
            elif 'litevae-s' in configs['vae_model']:
                img_latent_mean, seg_latent_mean = (
                    vae_model(img_rgb),
                    tiny_vae.encode(seg_rgb).latents,
                )
            elif 'imagenet' in configs['vae_model']:
                (_, _, img_latent_mean), seg_latent_mean = (
                    vae_model(img_rgb),
                    tiny_vae.encode(seg_rgb).latents,
                )
                # Get only 4 channels from img_latent_mean out of a total of 12
                img_latent_mean = img_latent_mean[:, :4, :, :]

            elif 'sd-vae' in configs['vae_model']:
                img_latent_mean, _ = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(img_rgb)), scale_factor)
                seg_latent_mean, _ = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(seg_rgb)), scale_factor)

            if configs['guidance_method']:
                    guidance_image = prepare_guidance(img_rgb, mode = configs['guidance_method'])
                    if configs['guidance_method'] == 'wavelet':
                        guidance_image = skff_module(guidance_image)

            out_latent_mean_dict = mapping_model(img_latent_mean, guidance_image) if configs['guidance_method'] else mapping_model(img_latent_mean.to(device))
            loss_Rec = configs['w_rec'] * mse_loss(out_latent_mean_dict['out'], seg_latent_mean)

            decoder_model = tiny_vae if configs['vae_model'] != 'tiny_vae' and 'sd-vae' not in configs['vae_model'] else vae_model

            pred_seg = vae_decode(decoder_model, out_latent_mean_dict['out'], scale_factor, sd_vae = True if 'sd-vae' in configs['vae_model'] else False)
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
    csv_path  = os.path.join(configs['snapshot_path'], 'results.csv')
    true_path = os.path.join(os.path.dirname(configs['pickle_file_path']), 'masks')

    pred_binary_path = save_seg_img_path
    pred_logits_path = save_seg_logits_path
    IMG_FORMAT       = '.png'

    name_list = sorted(os.listdir(save_seg_img_path))
    # Remove IMG_FORMAT from names
    name_list = [x.replace(IMG_FORMAT, '') for x in name_list]
    name_list = [x.replace('_binary', '') for x in name_list]

    dsc_list, iou_list, hd95_list = [], [], []
    ssim_list, ssim_region_list, ssim_object_list, ssim_combined_list = [], [], [], []


    for case_name in tqdm(name_list):
        seg_binary   = load_img(os.path.join(pred_binary_path, case_name + '_binary' +  IMG_FORMAT))
        seg_logits   = load_img(os.path.join(pred_logits_path, case_name + '_logits' + IMG_FORMAT))
        seg_true     = load_img(os.path.join(true_path, case_name + IMG_FORMAT))

        # Calculate all metrics
        results = all_metrics(seg_binary, seg_logits, seg_true)

        # Append all scores into respective list by inexing the results dict
        dsc_list.append(results['DSC'])
        iou_list.append(results['IoU'])
        hd95_list.append(results['HD95'])
        # fj
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
    logging.info('Time Taken: %d sec' % (time.time() - epoch_start_time))
    logging.info('\n')

# -----------------------------------------------------------------------#
if __name__ == "__main__":
    run_validator()
