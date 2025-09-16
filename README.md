# Wavelet-guided Generative Medical Segmentation (ICASSP 2026)

This is the official repository of **Wave-GMS**, a wavelet-enhanced extension of Generative Medical Segmentation (GMS).

[Paper](https://arxiv.org/pdf/2403.18198.pdf) | [Weights](ckpt)

---

## Updates
- **2025.01.10**: Wave-GMS released as an extension to GMS with LiteVAE integration.
- **2024.12.09**: Original GMS work accepted by AAAI 2025.
- **2024.05.13**: GMS code and model weights released.

---

## Introduction
We introduce **Wavelet-guided Generative Medical Segmentation (Wave-GMS)**, an extension of GMS that leverages **LiteVAE** and **wavelet-based structural guidance** for improved segmentation.  
Instead of relying only on the pre-trained Stable Diffusion VAE, Wave-GMS introduces:
- A lightweight **LiteVAE** encoder–decoder for efficiency.  
- A **wavelet guidance mechanism** (via SKFF fusion) to capture structural information.  
- Integration with existing latent mapping models (ResAttnUNet and SFT-UNet).  

Our design reduces computation while enhancing segmentation quality, particularly in medical datasets with structural textures.  
Extensive experiments across multiple public datasets show that **Wave-GMS achieves competitive Dice/IoU scores and superior SSIM-based similarity measures**.

---

## Overview of Wave-GMS
We combine LiteVAE with wavelet-guided latent mapping.  
- **LiteVAE**: lightweight encoder–decoder trained for medical segmentation tasks.  
- **Wavelet Guidance**: structural feature enhancement fused into the mapping model.  
- **Latent Mapping Model (LMM)**: maps from image latent to segmentation latent.  

![overview](assets/overview.png)

---

## Getting Started

### Environment Setup
We provide a [conda env file](environment.yaml) that contains all dependencies.  
You can create and activate the environment with:
```bash
conda env create -f environment.yaml
conda activate Wave-GMS
```
Or use the virtual environment:
```
python3 -m venv wavegms
source wavegms/bin/activate
pip install -r requirements.txt
```

### Prepare datasets
We evaluate GMS on five public datasets: [BUS](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php), [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and [Kvasir-Instrument](https://datasets.simula.no/kvasir-instrument/). The structure of the Dataset folder should be as follows:
```
Dataset/
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
├── kvasir-instrument
│   ├── kvasir_train_test_names.pkl
│   ├── images/
│   └── masks/
└── show_pkl.py
```
We provide the preprocessed **BUSI** and **Kvasir-Instrument** via [this link](https://emckclac-my.sharepoint.com/:f:/g/personal/k21066681_kcl_ac_uk/EmKNDZjEtg9EuBygBDyz4wIBKODtGJhzG2xdIy_NLf4VEA?e=whggsd), please download the dataset file and unzip it into the Dataset folder. For other datasets, please download them via the dataset websites and organize as the same structure. The `.pkl` file stores the train and test spilt for each dataset, you can run [`show_pkl.py`](Dataset/show_pkl.py) to show the content for each pkl file.


### Model Inference
We provide model weights for five dataset at [`ckpt`](ckpt) folder. Once all datasets are preprocessed, please run the following command for model inference:
```
sh valid.sh
```

The DSC, IOU, and predicted masks will be automatically saved.

### Model training
Please run the following command for model training:
```
sh train.sh
```

To change hyper-parameters (batchsize, learning rate, training epochs, etc.), please refer to the dataset training yaml file (e.g. [BUSI training yaml](configs/busi_train.yaml)). We train GMS on an NVIDIA A100 40G GPU with the batchsize set to 8. If you encounter the OOM problem, please try to decrease the batchsize. 

## Citation
If you use this code for your research, please consider citing our paper.
```
@article{wavegms2025,
  title={Wavelet-guided Generative Medical Segmentation},
  author={Talha Ahmed and Hassan Mohyuddin and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Acknowledgments
Thanks for the following code repositories: [TAESD](https://github.com/madebyollin/taesd) and [GMS](https://github.com/King-HAW/GMS)


