# ------------------------------------------------------------------------------#
#
# File name                 : load_pretrained_models.py
# Purpose                   : Unified helpers to download, load, and optionally
#                             freeze pretrained models (ckpt / safetensors / diffusers)
#                             from Hugging Face Hub.
# Usage                     : from utils.load_pretrained_models import load_pretrained_model
#                             vae = load_pretrained_model(AutoencoderKL, "stabilityai/sd-vae-ft-mse", dtype=torch.float16)
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module imports ----------------------------------#
import os
import logging
from typing import Optional, Union, Dict

import torch
from huggingface_hub import hf_hub_download

# Safetensors is optional
try:
	import safetensors.torch as safe_torch
	SAFETENSORS_AVAILABLE = True
except ImportError:
	SAFETENSORS_AVAILABLE = False
	logging.warning("[load_pretrained_models] safetensors not found; .safetensors files will fall back to torch.load().")


# --------------------------- Parameter utilities -----------------------------#
def freeze_model(model: torch.nn.Module, freeze: bool = True):
	"""Freezes or unfreezes a model's parameters."""
	for p in model.parameters():
		p.requires_grad = not freeze


def print_model_info(model: torch.nn.Module, name: str = "Model"):
	n_params = sum(p.numel() for p in model.parameters()) / 1e6
	n_trains = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
	print(f"[{name}] params: {n_params:.2f}M | trainable: {n_trains:.2f}M")


# --------------------------- State-dict loader -------------------------------#
def load_state_dict_from_ckpt(
	repo_id: str,
	filename: str,
	map_location: str = "cpu",
	use_safetensors: bool = False,
	sub_state_dict_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
	
	"""Downloads given ckpt/safetensors file and returns a state_dict."""
	ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.environ.get("HF_HOME", None))
	logging.info(f"[load_state_dict] downloaded → {ckpt_path}")

	if ckpt_path.endswith('.safetensors') or use_safetensors:
		if not SAFETENSORS_AVAILABLE:
			raise ImportError("safetensors is not installed. Install with `pip install safetensors`. ")
		state = safe_torch.load_file(ckpt_path, device=map_location)
	else:
		state = torch.load(ckpt_path, map_location=map_location)

	if sub_state_dict_key and sub_state_dict_key in state:
		state = state[sub_state_dict_key]
	return state


# --------------------------- Generic model loader ----------------------------#
def load_pretrained_model(
	model_cls,
	pretrained_name_or_path: str,
	*,
	dtype              : torch.dtype              = torch.float32,
	device             : Union[str, torch.device] = "cpu",
	freeze             : bool                     = True,
	subfolder          : Optional[str]            = None,
	use_safetensors    : bool                     = False,
	ckpt_filename      : Optional[str]            = None,
	sub_state_dict_key : Optional[str]            = None,
	map_location       : Optional[str]            = None,
):
	"""High-level loader for any diffusers model class or raw ckpt.

	Args:
		model_cls              : A diffusers model class with .from_pretrained()
		pretrained_name_or_path: HF repo (or local) when using .from_pretrained();
								 ignored if ckpt_filename is provided.
		dtype                  : torch.float32 / torch.float16 / torch.bfloat16
		device                 : CPU or CUDA device
		freeze                 : If True, sets requires_grad=False on all params
		subfolder              : For specific subfolders inside the repo (e.g. "vae")
		use_safetensors        : Load .safetensors file when ckpt_filename given
		ckpt_filename          : If provided, download that file instead of using
								 diffusers .from_pretrained().
		sub_state_dict_key     : Optional key inside ckpt dict (e.g. "state_dict")
		map_location           : Overrides device mapping for torch.load()
	Returns:
		Instantiated and optionally frozen model.
	"""
	device       = torch.device(device)
	map_location = map_location or device.type

	# ------------------- Case 1: Use diffusers.from_pretrained ---------------#
	if ckpt_filename is None:
		model = model_cls.from_pretrained(
			pretrained_name_or_path,
			subfolder   =   subfolder,
			torch_dtype =   dtype,
		)
		model.to(device)
	# ------------------- Case 2: Custom ckpt / safetensors -------------------#
	else:
		state_dict = load_state_dict_from_ckpt(
			repo_id             =   pretrained_name_or_path,
			filename            =   ckpt_filename,
			map_location        =   map_location,
			use_safetensors     =   use_safetensors,
			sub_state_dict_key  =   sub_state_dict_key,
		)
		# model = model_cls(**from_pretrained_kwargs)  # instantiate with default args
		model.load_state_dict(state_dict, strict=False)
		model.to(device, dtype = dtype)

	freeze_model(model, freeze)
	print_model_info(model, name = model.__class__.__name__)
	return model

# -------------------------------- End ----------------------------------------#