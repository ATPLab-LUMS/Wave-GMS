# ------------------------------------------------------------------------------#
#
# File name                 : load_ckpt.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# ckpt file in the hugging face cache directory. We can then access the ckpt file using the torch.load function.
# or we can get the state_dict from the ckpt file and then load the state_dict into the vae models

# pip install -U "huggingface_hub[hf_xet]"

import re, torch

from collections                              import OrderedDict
from diffusers                                import AutoencoderTiny as HF_TinyVAE
from huggingface_hub                          import hf_hub_download
from networks.novel.lite_vae.encoder          import LiteVAEEncoder
from networks.novel.lite_vae.decoder          import *
from networks.novel.lite_vae.litevae          import LiteVAE
from networks.novel.tiny_vae.autoencoder_tiny import AutoencoderTiny

def get_state_dict(ckpt_url = 'https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/768-v-ema.ckpt',
                   repo_id  = 'stabilityai/stable-diffusion-2',
                   filename = '768-v-ema.ckpt'):

    # Check out this

    # Download the checkpoint file
    ckpt_path_local = hf_hub_download(repo_id=repo_id, filename=filename)

    print(f"Checkpoint downloaded to: {ckpt_path_local}")

    # Load the state dictionary from the checkpoint file
    # This loads the dictionary of weights, but doesn't load the models themselves
    try:
        state_dict = torch.load(ckpt_path_local, map_location="cpu")
        print("Successfully loaded state dictionary.")
        # You can inspect the keys if you want to see what's inside
        # print(state_dict.keys()) # Uncomment to see the keys
    except Exception as e:
        print(f"Error loading state dictionary: {e}")

def get_tiny_autoencoder(device = 'cuda' if torch.cuda.is_available() else 'cpu',
                         mode = 'tiny', train = False, freeze = True,
                         residual_autoencoding = False):

    if not residual_autoencoding:
        print(f'Collecting AutoencoderTiny of mode {mode} from Diffusers Library')
        if mode == 'tiny':
            print('Downloading AutoencoderTiny Original')
            vae = HF_TinyVAE.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32, device_map = device)
        elif mode == 'hybrid':
            print('Downloading AutoencoderTiny Hybrid...')
            vae = HF_TinyVAE.from_pretrained("cqyan/hybrid-sd-tinyvae", torch_dtype=torch.float32, device_map = device)

    else:
        vae = load_residual_tiny_vae(device = device)
        vae = vae.to(device)

    if train:
        print('Training AutoencoderTiny')
        vae.train()
        return vae

    elif freeze:
        print('Freezing All params of AutoencoderTiny')
        for param in vae.parameters():
            param.requires_grad = False
        print('Freezing Complete...')
        vae.eval()
        return vae

def get_lite_vae(model_version = 'litevae-s', device: str = 'cuda' if torch.cuda.is_available() else 'cpu', dtype: torch.dtype = torch.float32, train = True, freeze = False) -> LiteVAE:

    base_model = LiteVAE(LiteVAEEncoder(model_version = model_version)).to(device=device, dtype=dtype).to(memory_format=torch.channels_last)
    if train:
        print('Training LiteVAE')
        base_model.train()
        return base_model

    elif freeze:
        print('Freezing All params of LiteVAE')
        for param in base_model.parameters():
            param.requires_grad = False
        print('Freezing Complete...')
        base_model.eval()
        return base_model

# ------------------------------------------------------------------
# Adjust these imports to match your repo layout
# ------------------------------------------------------------------y

def _remap_key(key: str) -> str:
    """Remaps encoder downsample layer keys to account for ResidualDownAE wrapping."""
    encoder_down_layers = {2, 6, 10}
    enc_match = re.match(r"(encoder\.layers\.(\d+))\.(weight|bias)", key)

    if enc_match:
        idx = int(enc_match.group(2))
        if idx in encoder_down_layers:
            return f"{enc_match.group(1)}.down.{enc_match.group(3)}"
    return key  # No change for other keys


@torch.no_grad()
def load_residual_tiny_vae(
    device: str = "cuda",
    dtype: torch.dtype = torch.float32
) -> torch.nn.Module:
    """Loads pretrained Tiny Autoencoder from HuggingFace and remaps to custom Residual model."""
    print("📦 Downloading AutoencoderTiny (HF: madebyollin/taesd)...")
    hf_vae = HF_TinyVAE.from_pretrained("madebyollin/taesd", torch_dtype=dtype, low_cpu_mem_usage=True)
    hf_state = hf_vae.state_dict()
    del hf_vae  # Free RAM immediately

    print("🔁 Remapping Keys for Residual Wrapper Compatibility...")
    remapped = OrderedDict((_remap_key(k), v) for k, v in hf_state.items())
    del hf_state  # Save memory

    print("🧠 Instantiating Residual AutoencoderTiny...")
    vae = AutoencoderTiny().to(device=device, dtype=dtype).to(memory_format=torch.channels_last)

    print("📥 Loading Remapped Weights...")
    missing, unexpected = vae.load_state_dict(remapped, strict=False)
    del remapped

    if missing or unexpected:
        print("⚠️  Load completed with unmatched keys:")
        print("  ❌ Missing   :", missing)
        print("  ⚠️ Unexpected:", unexpected)
    else:
        print("✅ State dict loaded successfully.")

    return vae