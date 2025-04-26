<div align="center">
<h1>Devil is in Details: Locality-Aware 3D Abdominal CT Volume Generation for Self-Supervised Organ Segmentation</h1>
</div>


This is the official implementation of **[Devil is in Details: Locality-Aware 3D Abdominal CT Volume Generation for Self-Supervised Organ Segmentation](https://dl.acm.org/doi/abs/10.1145/3664647.3680588) (ACM MM 2024)**

# Environment Setup
Create the conda environment by running:
```bash
conda env create -f environment.yml
```

# Data Preparation
## Data Download
The AbdomenCT-1K dataset can be downloaded from [this repository](https://github.com/JunMa11/AbdomenCT-1K).
## Data Preprocessing
Preprocessing scripts will be released soon.

# Training
## VQ-GAN
### Configuration
Set the model in `config/base_cfg.yaml` by specifying `model: vq_gan_3d`.
The detailed configuration for VQ-GAN training is provided in `config/model/vq_gan_3d.yaml`.
### Run
To start training VQ-GAN, run:
```bash
bash train_vqgan.sh
```

## Diffusion
### Configuration
Set the model in `config/base_cfg.yaml` by specifying `model: ddpm`.
The detailed configuration for diffusion model training is located in `config/model/ddpm.yaml`.
### Run
To start training the diffusion model, run:
```bash
bash train_ddpm.sh
```

## Checkpoints
Checkpoints for both VQ-GAN and the diffusion model can be downloaded from the links below:
| Model     | Checkpoint |
|-----------|------------|
| **VQ-GAN**    | [recon_loss=0.10.ckpt](https://huggingface.co/Ryann829/Lad/blob/main/recon_loss%3D0.10.ckpt) |
| **Diffusion** | [model-150.pt](https://huggingface.co/Ryann829/Lad/blob/main/model-150.pt) |

# Sampling
To generate samples using the trained model, run:
```bash
PL_TORCH_DISTRIBUTED_BACKEND=nccl NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env sample.py
```

# Citation
If you find this work helpful, please consider citing:
```
@inproceedings{wang2024devil,
  title={Devil is in Details: Locality-Aware 3D Abdominal CT Volume Generation for Self-Supervised Organ Segmentation},
  author={Wang, Yuran and Wan, Zhijing and Qiu, Yansheng and Wang, Zheng},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={10640--10648},
  year={2024}
}
```

# Acknowledgement
This project builds heavily upon the following repository:
* [Medical Diffusion](https://github.com/FirasGit/medicaldiffusion)
