# Wave-GMS: Lightweight Multi-Scale Generative Model for Medical Image Segmentation

!! Disclaimer !!
This work and much of the code has been inspired/adapted/borrowed from the [GMS](https://github.com/King-HAW/GMS).

This is the official repository of **Wave-GMS**: ightweight Multi-Scale Generative Model for Medical Image Segmentation.

Paper | Weights (Both available soon!)

---

## Updates
- **2025.09.17**: Wave-GMS released as an extension to GMS with Multi-Resolution Encoder Integration.
- **2024.12.09**: Original [GMS](https://github.com/King-HAW/GMS) work published AAAI 2025.
- **2024.05.13**: GMS code and model weights released.

---

## Introduction
We introduce **Wave-GMS**, an extension of GMS that leverages multi-scale representations and lightweight pre-trained models for improved segmentation. Instead of relying only on the pre-trained [Stable Diffusion VAE](https://github.com/Stability-AI/stablediffusion), Wave-GMS introduces:
- A multi-scale wavelet decomposition encoder with a frozen [Tiny-VAE](https://github.com/madebyollin/taesd) decoder, yielding a highly memory-efficient design.  
- An alignment loss in the latent space to facilitate cross-vae compatibility between multi-resolution encoder & Tiny-VAE.  
- Integration with existing latent mapping models ([ResAttnUNet_DS](https://github.com/King-HAW/GMS) and SFT_UNet_DS -- Their scripts will be made avaialble soon!).
  
Our model is highly memory efficient with a total trainable parameter count of ~2.60M and can be trained on low-end GPU's like RTX 3060 (12GB) or RTX 2080Ti (11GB). Extensive experiments across multiple public datasets show that **Wave-GMS achieves competitive Dice, IoU, and HD95 scores**.

---

## Overview of Wave-GMS
- It uses a trainable encoder multi-resolution encoder inspired by [Paper](https://arxiv.org/abs/2405.14477) to create high-quality latent representation from a multi-resolution Haar Wavelet decomposition of input image.
- The model leverages a compressed version of SD-VAE, Tiny-VAE, to generate latent representations of input image and segmentation mask.
- A Latent Mapping Model (LMM) learns the mapping from multi-resolution latent representation of input image to the corresponding mask representation
- Multi-resolution latents are aligned with Tiny-VAE’s latents to improve cross-VAE compatibility.
![overview](assets/overview.svg)

---

## Getting Started

### Environment Setup
We provide a requirements [file](requirements.txt) containing all dependencies. You can create and activate the virtual environment with:
```
python3 -m venv wavegms
source wavegms/bin/activate
pip install -r requirements.txt
```

### Prepare datasets
We evaluate GMS on four public datasets: [BUS](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php), [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and [Kvasir-Instrument](https://datasets.simula.no/kvasir-instrument/). The structure of the datasets folder should be as follows:
```
datasets/
├── bus
│   ├── bus_train_test_names.pkl
│   ├── images/
│   └── masks/
├── busi
│   ├── busi_train_test_names.pkl
│   ├── images/
│   └── masks/
├── ham10000
│   ├── ham10000_train_test_names.pkl
│   ├── images/
│   └── masks/
└── kvasir-instrument
    ├── kvasir_train_test_names.pkl
    ├── images/
    └── masks/
```
The `{dataset_name}_train_test_names.pkl` is a pickle file containing the train & test split for each dataset. It has two nested dictionaries (The first level dict is 'train' & 'test') and each of those has a sub dict with the key 'name_list'. The name list only covers the image names. The mask names can be found from a slight modification (if any) in the image names like '_segmentation' for HAM10000, etc. 
The preprocessed **BUSI** and **Kvasir-Instrument** can be through the original GMS repository. Please download the dataset file and unzip it into the datasets folder. For other datasets, please download them via the dataset websites and organize as the same structure.

### Model Inference
We (will soon) provide the inference script and model weights for four datasets at [`ckpt`](ckpt) folder. Once all datasets are preprocessed, please run the following inference command:
```
sh valid.sh
```
The Dice, IoU, HD95 (calculated by script [`utils/metrics.py`] - available soon), and predicted masks will be automatically saved. The predicted masks (binary & logits) will be saved in the folder `./ckpt/experiment_name/epochs_{epoch_num}/predicted_masks_{dataset_name}` while the csv file containing the metrics against each patient in the dataset will be saved in `./ckpt/experiment_name/epochs_{epoch_num}/valid_results_{dataset_name}`

### Model training
We (will soon) provide the training script. Please run the following command for model training:
```
sh train.sh
```
For hyperparameter-tuning, please refer to the dataset training yaml file (e.g., [BUSI training yaml](configs/busi_train.yaml)). We train Wave-GMS on an RTX 3060 GPU (12 GB) with a batch size of 12. If you encounter the OOM problem, please try to decrease the batch size. 

## Citation
If you use this code for your research, please consider citing this github page.

## Acknowledgments
We thank the following code repositories: [TAESD](https://github.com/madebyollin/taesd) and [GMS](https://github.com/King-HAW/GMS).
