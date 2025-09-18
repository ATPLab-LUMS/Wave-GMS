# Wave-GMS: Lightweight Multi-Scale Generative Model for Medical Image Segmentation

⚠️ **Disclaimer**  
This work and much of the code has been adapted or extended from the original [GMS repository](https://github.com/King-HAW/GMS).

This is the official repository of **Wave-GMS**: a lightweight multi-scale generative model for medical image segmentation.

**Paper | Weights (Coming Soon!)**

---

## Updates
- **2025.09.17**: Wave-GMS released as an extension to GMS with multi-resolution encoder integration.  
- **2024.12.09**: Original [GMS](https://github.com/King-HAW/GMS) accepted at AAAI 2025.  
- **2024.05.13**: GMS code and model weights released.  

---

## Introduction
We introduce **Wave-GMS**, an extension of GMS that leverages multi-scale representations and lightweight pre-trained models for improved segmentation. Instead of relying solely on the pre-trained [Stable Diffusion VAE](https://github.com/Stability-AI/stablediffusion), Wave-GMS introduces:
- A multi-scale wavelet decomposition encoder coupled with a frozen [Tiny-VAE](https://github.com/madebyollin/taesd) decoder, yielding a highly memory-efficient design.  
- An alignment loss in the latent space to ensure compatibility between the multi-resolution encoder and Tiny-VAE.  
- Integration with existing latent mapping models ([ResAttnUNet_DS](https://github.com/King-HAW/GMS) and SFT_UNet_DS — scripts will be made available soon).  

Our model is highly memory-efficient, with only ~2.6M trainable parameters, and can be trained on low_end GPUs such as the RTX 3060 (12GB) or RTX 2080Ti (11GB). Extensive experiments on multiple public datasets demonstrate that **Wave-GMS achieves competitive Dice, IoU, and HD95 scores** while being lightweight and efficient.  

---

## Overview of Wave-GMS
- A trainable multi-resolution encoder, inspired by [this work](https://arxiv.org/abs/2405.14477), creates high-quality latent representations from a Haar wavelet decomposition of the input image.  
- A compressed distilled version of SD-VAE (Tiny-VAE) generates latent representations of both the input image and segmentation mask.  
- A Latent Mapping Model (LMM) learns the mapping from the multi-resolution latent space of the input image to the corresponding mask representation.  
- Multi-resolution latents are aligned with Tiny-VAE’s latents to improve cross-VAE compatibility.  

![overview](assets/overview.svg)

---

## Getting Started

### Environment Setup
We provide a [requirements file](requirements.txt) containing all dependencies. You can create and activate a virtual environment with:

```bash
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
Each `{dataset_name}_train_test_names.pkl` contains the train/test splits in the form of nested dictionaries (train and test), each with a name_list key. These lists cover image filenames. Masks follow the same names with small modifications (e.g., _segmentation suffix in HAM10000).
The preprocessed **BUSI** and **Kvasir-Instrument** datasets can be obtained through the original GMS repository. Please download the dataset file and unzip it into the datasets folder. For other datasets, please download them via the dataset websites and organize as the same structure.

### Model Inference
We (will soon) provide the inference script and model weights for four datasets at [`ckpt/provided_models`](ckpt) folder. Once all datasets are preprocessed, please run the following inference command:
```
sh valid.sh
```
Metrics (Dice, IoU, HD95 — computed using [`utils/metrics.py`], coming soon) and predicted masks will be automatically saved.

- **Predicted masks (binary & logits):**  
  `./ckpt/experiment_name/epochs_{epoch_num}/predicted_masks_{dataset_name}`

- **Metrics CSV file:**  
  `./ckpt/experiment_name/epochs_{epoch_num}/valid_results_{dataset_name}`

### Model training
We (will soon) provide the training script. Please run the following command for model training:
```
sh train.sh
```
For hyperparameter-tuning, please refer to the dataset training yaml file (e.g., [BUSI training yaml](configs/busi_train.yaml)). We train Wave-GMS on an RTX 3060 GPU (12 GB) with a batch size of 12. If you encounter the OOM problem, please try to decrease the batch size. 

## Citation
If you use this code for your research, please consider citing this github page:

@misc{ATPLab-LUMS_2025,
  title   = {ATPLab-lums/wave-GMS: [submitted to ICASSP 2026] official repository of Wave-GMS: Lightweight Multi-Scale Generative Model for Medical Image Segmentation},
  url     = {https://github.com/ATPLab-LUMS/Wave-GMS/tree/main},
  journal = {GitHub},
  author  = {ATPLab-LUMS and Talha Ahmed},
  year    = {2025},
  month   = {Sep}
}

## Acknowledgments
We thank the following code repositories: [TAESD](https://github.com/madebyollin/taesd) and [GMS](https://github.com/King-HAW/GMS).
