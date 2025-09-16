# ------------------------------------------------------------------------------#
#
# File name                 : valid.py
# Purpose                   : Validation script for evaluating latent mapping model.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : Sep 17, 2025
# Note                      : Adapted from [https://github.com/King-HAW/GMS]
# ------------------------------------------------------------------------------#

# ------------------------------- Basic Packages -------------------------------#
import os, time, yaml, torch, logging, argparse

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader

# ------------------------------- Own Packages --------------------------------#
from data.image_dataset import Image_Dataset

from utils.tools import *
from utils.get_logger import open_log
from utils.metrics import all_metrics
from utils.instantiate_model import get_tiny_autoencoder, get_lite_vae

from modules.latent_mapping_model import ResAttnUNet_DS
from modules.sft_lmm import *
from modules.guidance import *

# ---------------------------- Helper Functions -------------------------------#
def save_binary_and_logits(x_logits, x_binary, name, save_seg_img_path, save_seg_logits_path, IMG_FORMAT=".png"):
    """Save predicted binary and logits masks."""
    x_binary.save(os.path.join(save_seg_img_path, name + "_binary" + IMG_FORMAT))
    x_logits = (x_logits * 255).astype(np.uint8)
    x_logits = Image.fromarray(x_logits)
    x_logits.save(os.path.join(save_seg_logits_path, name + "_logits" + IMG_FORMAT))


def load_img(path, img_size=224, dtype_resize=np.float32):
    """Load grayscale mask, resize and normalize to [0,1]."""
    image = Image.open(path).convert("L").resize((img_size, img_size), resample=Image.NEAREST)
    return np.array(image).astype(dtype_resize) / 255.0


def vae_decode(vae_model, pred_mean, scale_factor):
    """Decode latent representation into segmentation prediction."""
    z = 1.0 / scale_factor * pred_mean
    pred_seg = vae_model.decode(z).sample
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True)  # → 1 channel
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0)
    return pred_seg


def arg_parse() -> argparse.ArgumentParser.parse_args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/config.yaml",
        type=str,
        help="load the config file",
    )
    return parser.parse_args()


# ------------------------------- Main Validator -------------------------------#
def run_validator() -> None:
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)

    # Paths
    save_seg_img_path = os.path.join(configs["save_seg_img_path"], "binary")
    save_seg_logits_path = os.path.join(configs["save_seg_img_path"], "logits")
    configs["log_path"] = os.path.join(configs["snapshot_path"], "logs")

    os.makedirs(configs["snapshot_path"], exist_ok=True)
    os.makedirs(save_seg_img_path, exist_ok=True)
    os.makedirs(save_seg_logits_path, exist_ok=True)
    os.makedirs(configs["log_path"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU
    if torch.cuda.is_available():
        gpus = ",".join([str(i) for i in configs["GPUs"]])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Seed
    seed_reproducer(configs["seed"])

    # Logger
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # ------------------------------ Dataset --------------------------------#
    valid_dataset = Image_Dataset(configs["pickle_file_path"], stage="test")

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    # ------------------------------- Models --------------------------------#
    skff_module = None
    if configs["guidance_method"]:
        guidance_channels_dict = {"edge": 3, "wavelet": 3, "dino": 384}
        mapping_model = SFT_UNet_DS(
            in_channels=configs["in_channel"],
            out_channels=configs["out_channels"],
            guidance_channels=guidance_channels_dict[configs["guidance_method"]],
        ).to(device)
        if configs["guidance_method"] == "wavelet":
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

    vae_train = False
    if configs["vae_model"] == "tiny_vae":
        logging.info("Initializing TinyVAE")
        vae_model = get_tiny_autoencoder(train=False, residual_autoencoding=False)
    else:
        logging.info("Initializing LiteVAE")
        tiny_vae = get_tiny_autoencoder(train=False, residual_autoencoding=False)
        vae_model = get_lite_vae(model_version=configs["vae_model"], train=False, freeze=True)
    vae_model.eval()

    scale_factor = 1.0

    # Load weights
    if vae_train and skff_module is None:
        mapping_model, vae_model, _ = load_checkpoint(mapping_model, configs["model_weight"], vae_model=vae_model, vae_model_load=True)
    elif vae_train and skff_module is not None:
        mapping_model, vae_model, skff_module = load_checkpoint(mapping_model, configs["model_weight"], vae_model=vae_model, vae_model_load=True, skff_model=skff_module, skff_model_load=True)
    elif not vae_train and skff_module is not None:
        mapping_model, vae_model, skff_module = load_checkpoint(mapping_model, configs["model_weight"], vae_model=vae_model, vae_model_load=False, skff_model=skff_module, skff_model_load=True)
    else:
        mapping_model, _, _ = load_checkpoint(mapping_model, configs["model_weight"])

    mapping_model.eval()
    vae_model.eval()

    # --------------------------- Validation Loop ---------------------------#
    mse_loss = torch.nn.MSELoss(reduction="mean")
    epoch_start_time = time.time()

    name_list, T_loss_valid = [], []
    for batch_data in tqdm(valid_dataloader, desc="Valid: "):
        img_rgb = batch_data["img"].to(device) / 255.0

        if configs["vae_model"] in ["tiny_vae", "sd-vae"]:
            img_rgb = 2.0 * img_rgb - 1.0

        seg_raw = batch_data["seg"].to(device).permute(0, 3, 1, 2) / 255.0
        seg_rgb = 2.0 * seg_raw - 1.0
        name = batch_data["name"][0]
        name_list.append(name)

        with torch.no_grad():
            if configs["vae_model"] == "tiny_vae":
                img_latent_mean, seg_latent_mean = vae_model.encode(img_rgb).latents, vae_model.encode(seg_rgb).latents
            else:
                img_latent_mean, seg_latent_mean = vae_model(img_rgb), tiny_vae.encode(seg_rgb).latents

            if configs["guidance_method"]:
                guidance_image = prepare_guidance(img_rgb, mode=configs["guidance_method"])
                if configs["guidance_method"] == "wavelet":
                    guidance_image = skff_module(guidance_image)

            out_latent_mean_dict = mapping_model(img_latent_mean, guidance_image) if configs["guidance_method"] else mapping_model(img_latent_mean)

            loss_Rec = configs["w_rec"] * mse_loss(out_latent_mean_dict["out"], seg_latent_mean)
            pred_seg = vae_decode(tiny_vae if configs["vae_model"] != "tiny_vae" else vae_model, out_latent_mean_dict["out"], scale_factor)
            pred_seg = pred_seg.repeat(1, 3, 1, 1)

            # Save predictions
            x_logits = rearrange(pred_seg.squeeze().cpu().numpy(), "c h w -> h w c")
            x_binary = np.where(x_logits > 0.5, 1, 0) * 255.0
            x_binary = Image.fromarray(x_binary.astype(np.uint8))
            save_binary_and_logits(x_logits, x_binary, name, save_seg_img_path, save_seg_logits_path)

            T_loss_valid.append(loss_Rec.item())

    T_loss_valid = np.mean(T_loss_valid)
    logging.info(f"Valid:\nloss: {T_loss_valid:.4f}")

    # --------------------------- Metrics & CSV -----------------------------#
    csv_path = os.path.join(configs["snapshot_path"], "results.csv")
    true_path = os.path.join(os.path.dirname(configs["pickle_file_path"]), "masks")

    pred_binary_path, pred_logits_path = save_seg_img_path, save_seg_logits_path
    IMG_FORMAT = ".png"

    name_list = sorted(os.listdir(save_seg_img_path))

    dsc_list, iou_list, hd95_list = [], [], []
    ssim_list, ssim_region_list, ssim_object_list, ssim_combined_list = [], [], [], []

    for case_name in tqdm(name_list):
        seg_binary = load_img(os.path.join(pred_binary_path, case_name + "_binary" + IMG_FORMAT))
        seg_logits = load_img(os.path.join(pred_logits_path, case_name + "_logits" + IMG_FORMAT))
        seg_true = load_img(os.path.join(true_path, case_name + IMG_FORMAT))

        results = all_metrics(seg_binary, seg_logits, seg_true)
        dsc_list.append(results["DSC"])
        iou_list.append(results["IoU"])
        hd95_list.append(results["HD95"])
        ssim_list.append(results["SSIM"])
        ssim_region_list.append(results["SSIM_region"])
        ssim_object_list.append(results["SSIM_object"])
        ssim_combined_list.append(results["SSIM_combined"])

    # Append mean & std
    name_list.extend(["Avg", "Std"])
    dsc_list.extend([np.mean(dsc_list), np.std(dsc_list, ddof=1)])
    iou_list.extend([np.mean(iou_list), np.std(iou_list, ddof=1)])
    hd95_list.extend([np.mean(hd95_list), np.std(hd95_list, ddof=1)])
    ssim_list.extend([np.mean(ssim_list), np.std(ssim_list, ddof=1)])
    ssim_region_list.extend([np.mean(ssim_region_list), np.std(ssim_region_list, ddof=1)])
    ssim_object_list.extend([np.mean(ssim_object_list), np.std(ssim_object_list, ddof=1)])
    ssim_combined_list.extend([np.mean(ssim_combined_list), np.std(ssim_combined_list, ddof=1)])

    pd.DataFrame({
        "Name": name_list,
        "DSC": dsc_list,
        "IoU": iou_list,
        "HD95": hd95_list,
        "SSIM": ssim_list,
        "SSIM_region": ssim_region_list,
        "SSIM_object": ssim_object_list,
        "SSIM_combined": ssim_combined_list,
    }).to_csv(csv_path, index=False)

    logging.info(f"DSC: {dsc_list[-2]:.4f}, IoU: {iou_list[-2]:.4f}, HD95: {hd95_list[-2]:.2f}")
    logging.info("Time Taken: %d sec" % (time.time() - epoch_start_time))


# ------------------------------- Entry Point ---------------------------------#
if __name__ == "__main__":
    run_validator()