import torch, argparse, yaml, logging, time, os
import numpy as np
from tqdm                         import tqdm
from torch.utils.data             import DataLoader
from monai.losses.dice            import DiceLoss
from data.image_dataset           import Image_Dataset
from tools.utils                  import *
from tools.get_logger             import open_log
from tools.load_ckpt              import *
from tools.lr_scheduler           import LinearWarmupCosineAnnealingLR
from tools.metrics                import *
from modules.latent_mapping_model import ResAttnUNet_DS
from modules.models.distributions import DiagonalGaussianDistribution
from modules.sft_lmm              import *
from modules.guidance             import *
from tensorboardX                 import SummaryWriter

def get_multi_loss(criterion, out_dict, label, is_ds=True, key_list=None):
    keys = key_list if key_list is not None else list(out_dict.keys())
    if is_ds:
        multi_loss = sum([criterion(out_dict[key], label) for key in keys])
    else:
        multi_loss = criterion(out_dict["out"], label)
    return multi_loss

def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
        
    else:
        raise NotImplementedError(
            f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
        )
    return scale_factor * mean, logvar

def vae_decode(vae_model, pred_mean, scale_factor):
    z = 1.0 / scale_factor * pred_mean
    pred_seg = vae_model.decode(z).sample
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True)
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0)  # (B, 1, H, W)
    return pred_seg

def arg_parse() -> argparse.ArgumentParser.parse_args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/kvasir-instrument_train.yaml",
        type=str,
        help="load the config file",
    )
    args   = parser.parse_args()
    return args

def run_trainer() -> None:
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    # time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    configs["snapshot_path"] = os.path.join(
        configs["snapshot_path"],
        'epochs_' + str(configs["epochs"]),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs["log_path"] = os.path.join(configs["snapshot_path"], "logs")

    # Output folder and save fig folder
    os.makedirs(configs["snapshot_path"], exist_ok=True)
    os.makedirs(configs["log_path"], exist_ok=True)

    # Set GPU ID
    if torch.cuda.is_available():
        gpus = ",".join([str(i) for i in configs["GPUs"]])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Fix seed (for repeatability)
    seed_reproducer(configs["seed"])

    # Open log file
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # Define summary writer
    writer = SummaryWriter(configs["log_path"])
    ds_list = ["level2", "level1", "out"]

    # Get data loader
    train_dataset = Image_Dataset(configs["pickle_file_path"], stage="train", excel = False)
    valid_dataset = Image_Dataset(configs["pickle_file_path"], stage="test", excel = False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=configs["batch_size"],
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    # Define modules
    skff_module = None
    if configs['guidance_method']:
        guidance_channels_dict = {'edge': 3, 'wavelet': 3, 'dino': 384}

        mapping_model = SFT_UNet_DS(in_channels       = configs['in_channel'],
                                    out_channels      = configs['out_channels'],
                                    guidance_channels = guidance_channels_dict[configs['guidance_method']]).to(device)

        if configs['guidance_method'] == 'wavelet':
            skff_module = SKFF().to(device)
            skff_module.train()
    else:
        mapping_model = ResAttnUNet_DS(
            in_channel=configs["in_channel"],
            out_channels=configs["out_channels"],
            num_res_blocks=configs["num_res_blocks"],
            ch=configs["ch"],
            ch_mult=configs["ch_mult"],
            ).to(device)

    mapping_model.train()
    param_groups = list(mapping_model.parameters())

    # Getting tiny-vae (with residual_autoencoding) default: frozen and eval
    vae_train = True
    if configs['vae_model'] == 'tiny_vae':
        logging.info("Initializing TinyVAE")
        vae_model = get_tiny_autoencoder(train = False)
    else:
        logging.info("Initializing LiteVAE")
        tiny_vae  = get_tiny_autoencoder(train = False) # for the segmentation latent and decoding at the end.
        vae_model = get_lite_vae(train = vae_train, model_version = configs['vae_model'])

    scale_factor = 1.0 # default
    # Define optimizers

    if vae_train:
        param_groups += list(vae_model.parameters())
        logging.info("Training both mapping model and VAE model")

    if skff_module is not None:
        param_groups += list(skff_module.parameters())
        logging.info('Training SKFF Module')
        logging.info(f"SKFF Module trainable params: {count_params(skff_module)} out of {sum(p.numel() for p in skff_module.parameters())}")

    logging.info(f"Mapping Model trainable params: {count_params(mapping_model)} out of {sum(p.numel() for p in mapping_model.parameters())}")
    logging.info(f"VAE Model trainable params: {count_params(vae_model)} out of {sum(p.numel() for p in vae_model.parameters())}")

    optimizer = torch.optim.AdamW(param_groups, lr=configs["lr"])
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=5, max_epochs=configs["epochs"]
    )

    # Define loss functions
    mse_loss = torch.nn.MSELoss(reduction="mean")
    dice_loss = DiceLoss() # for the final predicted mask

    # For Tensorboard Visualization
    iter_num = 0
    best_valid_loss = np.inf
    best_valid_loss_rec = np.inf
    best_valid_dice = 0
    best_valid_dice_epoch = 0
    best_valid_loss_dice = np.inf

    # Network training
    for epoch in range(1, configs["epochs"] + 1):
        epoch_start_time = time.time()
        mapping_model.train()
        vae_model.train() if vae_train else vae_model.eval()
        tiny_vae.eval() if configs['vae_model'] != 'tiny_vae' else None

        T_loss = []
        T_loss_Rec = []
        T_loss_Dice = []
        T_loss_valid = []
        T_loss_Rec_valid = []
        T_loss_Dice_valid = []
        T_Dice_valid = []

        ### Training phase
        for batch_data in tqdm(train_dataloader, desc="Train: "):
            # Note: no change needed regarding the segmentation as it is being used by tiny vae.
            img_rgb = batch_data['img'].to(device)
            img_rgb = img_rgb / 255.0

            if configs['vae_model'] == 'tiny_vae':
                img_rgb = 2. * img_rgb - 1.

            seg_raw = batch_data['seg'].to(device)
            seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2. * seg_raw - 1.
            # of pred_seg gotten from the decoder. Same thing in Validation
            seg_img = torch.mean(seg_raw, dim=1, keepdim=True).to(device)
            name = batch_data["name"]

            if configs['vae_model'] == 'tiny_vae':
                img_latent_mean_aug, seg_latent_mean = (
                    vae_model.encode(img_rgb).latents,
                    vae_model.encode(seg_rgb).latents,
                )
            else:
                img_latent_mean_aug, seg_latent_mean = (
                    vae_model(img_rgb),
                    tiny_vae.encode(seg_rgb).latents,
                )

            # get guidance image for the LMM
            if configs['guidance_method'] and configs['guidance_method'] != 'dino':
                with torch.no_grad():
                    guidance_image = prepare_guidance(img_rgb, mode = configs['guidance_method'])

                if configs['guidance_method'] == 'wavelet' and skff_module is not None:
                    guidance_image = skff_module(guidance_image) # (B, 3, 112, 112)

            elif configs['guidance_method'] and configs['guidance_method'] == 'dino':
                guidance_image = None

            # latent matching
            out_latent_mean_dict = mapping_model(img_latent_mean_aug, guidance_image) if configs['guidance_method'] else mapping_model(img_latent_mean_aug)
            loss_Rec = configs["w_rec"] * get_multi_loss(
                mse_loss,
                out_latent_mean_dict,
                seg_latent_mean,
                is_ds=True,
                key_list=ds_list,
            )

            # image matching
            # was on different levels (see line 228 - 233 of latent_mapping_model.py)
            pred_seg_dict = {}
            for level_name in ds_list:
                pred_seg_dict[level_name] = vae_decode(
                    tiny_vae if configs['vae_model'] != 'tiny_vae' else vae_model, out_latent_mean_dict[level_name], scale_factor
                )

            loss_Dice = configs["w_dice"] * get_multi_loss(
                dice_loss,
                pred_seg_dict,
                seg_img,
                is_ds=True,
                key_list=ds_list,
            )

            loss = loss_Rec + loss_Dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            if iter_num % 10 == 0:
                writer.add_scalar("loss/loss", loss, iter_num)
                writer.add_scalar("loss/loss_Rec", loss_Rec, iter_num)
                writer.add_scalar("loss/loss_Dice", loss_Dice, iter_num)

            T_loss.append(loss.item())
            T_loss_Rec.append(loss_Rec.item())
            T_loss_Dice.append(loss_Dice.item())

        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        T_loss = np.mean(T_loss)
        T_loss_Rec = np.mean(T_loss_Rec)
        T_loss_Dice = np.mean(T_loss_Dice)

        logging.info("Train:")
        logging.info(
            "loss: {:.4f}, loss_Rec: {:.4f}, loss_Dice: {:.4f}".format(
                T_loss, T_loss_Rec, T_loss_Dice
            )
        )

        writer.add_scalar("train/loss", T_loss, epoch)
        writer.add_scalar("train/loss_Rec", T_loss_Rec, epoch)
        writer.add_scalar("train/loss_Dice", T_loss_Dice, epoch)

        ### Validation phase
        ### Also note that the validation set is the same as the test set in the inference/valid.py file.
        for batch_data in tqdm(valid_dataloader, desc="Valid: "):
            img_rgb = batch_data["img"].to(device)
            img_rgb = img_rgb / 255.0
            if configs['vae_model'] == 'tiny_vae':
                img_rgb = 2.0 * img_rgb - 1.0


            seg_raw = batch_data["seg"].to(device)
            seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2. * seg_raw -1.
            seg_img = torch.mean(seg_raw, dim = 1, keepdim = True)
            name = batch_data['name']

            mapping_model.eval()
            vae_model.eval()
            tiny_vae.eval() if configs['vae_model'] != 'tiny_vae' else None
            if skff_module is not None: skff_module.eval()

            with torch.no_grad():
                if configs['vae_model'] == 'tiny_vae':
                    img_latent_mean, seg_latent_mean = (
                        vae_model.encode(img_rgb).latents,
                        vae_model.encode(seg_rgb).latents,
                    )
                else:
                    img_latent_mean, seg_latent_mean = (
                        vae_model(img_rgb),
                        tiny_vae.encode(seg_rgb).latents,
                    )

                if configs['guidance_method'] and configs['guidance_method'] != 'dino':
                    with torch.no_grad():
                        guidance_image = prepare_guidance(img_rgb, mode = configs['guidance_method'])

                    if configs['guidance_method'] == 'wavelet' and skff_module is not None:
                        guidance_image = skff_module(guidance_image) # (B, 3, 112, 112)

                elif configs['guidance_method'] and configs['guidance_method'] == 'dino':
                    guidance_image = None
                else:
                    guidance_image = None

                # latent matching
                out_latent_mean_dict = mapping_model(img_latent_mean, guidance_image) if configs['guidance_method'] else mapping_model(img_latent_mean)

                pred_seg = vae_decode(
                    tiny_vae if configs['vae_model'] != 'tiny_vae' else vae_model, out_latent_mean_dict["out"], scale_factor
                )

                loss_Rec = configs["w_rec"] * mse_loss(
                    out_latent_mean_dict["out"], seg_latent_mean
                )
                loss_Dice = configs["w_dice"] * dice_loss(pred_seg, seg_img)

                loss = loss_Rec + loss_Dice

                # calc dice
                pred_seg = pred_seg.cpu()
                reduce_axis = list(range(1, len(seg_img.shape)))

                intersection = torch.sum(seg_img.cpu() * pred_seg, dim=reduce_axis)
                y_o = torch.sum(seg_img.cpu(), dim=reduce_axis)
                y_pred_o = torch.sum(pred_seg, dim=reduce_axis)
                denominator = y_o + y_pred_o
                dice_raw = (2.0 * intersection) / denominator
                dice_value = dice_raw.mean()

                T_Dice_valid.append(dice_value.item())
                T_loss_valid.append(loss.item())
                T_loss_Rec_valid.append(loss_Rec.item())
                T_loss_Dice_valid.append(loss_Dice.item())

        T_Dice_valid = np.mean(T_Dice_valid)
        T_loss_valid = np.mean(T_loss_valid)
        T_loss_Rec_valid = np.mean(T_loss_Rec_valid)
        T_loss_Dice_valid = np.mean(T_loss_Dice_valid)

        writer.add_scalar("valid/dice", T_Dice_valid, epoch)
        writer.add_scalar("valid/loss", T_loss_valid, epoch)
        writer.add_scalar("valid/loss_Rec", T_loss_Rec_valid, epoch)
        writer.add_scalar("valid/loss_Dice", T_loss_Dice_valid, epoch)

        logging.info("Valid:")
        logging.info(
            "loss: {:.4f}, loss_Rec: {:.4f}, loss_Dice: {:.4f}, Dice: {:.4f}".format(
                T_loss_valid, T_loss_Rec_valid, T_loss_Dice_valid, T_Dice_valid
            )
        )

        if T_Dice_valid > best_valid_dice:
            save_name = f"best_valid_dice_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], skff_model = skff_module, skff_model_save = skff_module is not None)
            best_valid_dice = T_Dice_valid
            best_valid_dice_epoch = epoch
            logging.info("Save best valid Dice !")

        if T_loss_valid < best_valid_loss:
            save_name = f"best_valid_loss_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], skff_model = skff_module, skff_model_save = skff_module is not None)
            best_valid_loss = T_loss_valid
            logging.info("Save best valid Loss All !")

        if T_loss_Rec_valid < best_valid_loss_rec:
            save_name = f"best_valid_loss_rec_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], skff_model = skff_module, skff_model_save = skff_module is not None)
            best_valid_loss_rec = T_loss_Rec_valid
            logging.info("Save best valid Loss Rec !")

        if T_loss_Dice_valid < best_valid_loss_dice:
            save_name = f"best_valid_loss_dice_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], skff_model = skff_module, skff_model_save = skff_module is not None)
            best_valid_loss_dice = T_loss_Dice_valid
            logging.info("Save best valid Loss Dice !")

        if epoch % configs["save_freq"] == 0:
            save_name = "{}_epoch_{:0>4}.pth".format("latent_mapping_model", epoch)
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], skff_model = skff_module, skff_model_save = skff_module is not None)

        logging.info("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
        logging.info(
            "epoch %d / %d \t Time Taken: %d sec"
            % (epoch, configs["epochs"], time.time() - epoch_start_time)
        )
        logging.info(
            "best valid dice: {:.4f} at epoch: {}".format(
                best_valid_dice, best_valid_dice_epoch
            )
        )
        logging.info("\n")

    writer.close()


if __name__ == "__main__":
    run_trainer()
