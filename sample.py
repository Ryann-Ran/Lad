import torch
import numpy as np
import argparse
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True
import os
import sys
torch.cuda.set_device(0)
device = 'cuda:0'
sys.path.append('../')
from vq_gan_3d.model.vqgan import VQGAN
from train.get_dataset import get_dataset
from ddpm import Unet3D, GaussianDiffusion, Trainer
from hydra import initialize, compose
import glob as glob
import torch.distributed as dist

dist.init_process_group(backend='nccl')

parser = argparse.ArgumentParser()
parser.add_argument('--ddpm', default="checkpoints/ddpm/ABDOMENCT1K/model-150.pt", type=str)
parser.add_argument('--vqgan', default="checkpoints/vq_gan/ABDOMENCT1K/ABDOMENCT1K/lightning_logs/version_0/checkpoints/epoch\=124-step\=100000-10000-train/recon_loss\=0.10.ckpt", type=str)
parser.add_argument('--dir', default="results", type=str)
parser.add_argument('--batchsize', default="1", type=int)
parser.add_argument('--mask-source-path', default="data/ABDOMENCT1K/mask", type=str)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()


args.local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(args.local_rank)
torch.backends.cudnn.enabled = False

VQGAN_CHECKPOINT = args.vqgan
with initialize(config_path="./config/"):
    cfg=compose(config_name="base_cfg.yaml", overrides=[
        "model=ddpm",
        "dataset=abdomenct1k",
        f"model.vqgan_ckpt={VQGAN_CHECKPOINT}"
        ])

VQGAN_CHECKPOINT = VQGAN_CHECKPOINT.replace("\\", "")
vqgan = VQGAN.load_from_checkpoint(VQGAN_CHECKPOINT)
vqgan = vqgan.to(device)
vqgan.eval()



model = Unet3D(
    dim=cfg.model.diffusion_img_size,
    dim_mults=cfg.model.dim_mults,
    channels=cfg.model.diffusion_num_channels,
).cuda()

diffusion = GaussianDiffusion(
    model,
    vqgan_ckpt=cfg.model.vqgan_ckpt,
    image_size=cfg.model.diffusion_img_size,
    num_frames=cfg.model.diffusion_depth_size,
    channels=cfg.model.diffusion_num_channels,
    timesteps=cfg.model.timesteps,
    loss_type=cfg.model.loss_type,
).cuda()

train_dataset, val_dataset, _ = get_dataset(cfg)

trainer = Trainer(
    diffusion,
    cfg=cfg,
    dataset=train_dataset,
    train_batch_size=cfg.model.batch_size,
    save_and_sample_every=cfg.model.save_and_sample_every,
    train_lr=cfg.model.train_lr,
    train_num_steps=cfg.model.train_num_steps,
    gradient_accumulate_every=cfg.model.gradient_accumulate_every,
    ema_decay=cfg.model.ema_decay,
    amp=cfg.model.amp,
    num_sample_rows=cfg.model.num_sample_rows,
    results_folder=cfg.model.results_folder,
    num_workers=cfg.model.num_workers
)
trainer.load(args.ddpm, map_location='cuda:0')
trainer.ema_model.eval()

save_dir = args.dir
os.makedirs(save_dir, exist_ok=True)

mask_path_list = glob.glob(args.mask_source_path + '/*.npy')
mask_path_list = sorted(mask_path_list)

batch_size = args.batchsize
for i in range(len(mask_path_list) // batch_size):
    save_path_list = []
    mask = None
    
    for j in range(batch_size):
        name = mask_path_list[i * batch_size + j].split("/")[-1].split(".")[0]
        save_path = save_dir + f"/{name}.npy"
        save_path_list.append(save_path)
        if j == 0:
            mask = torch.from_numpy(np.load(mask_path_list[i * batch_size])).float().unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.cat((mask, torch.from_numpy(np.load(mask_path_list[i * batch_size + j])).float().unsqueeze(0).unsqueeze(0)),dim=0)
                
    with torch.no_grad():
        mask = mask.cuda()
        assert mask.shape  == (batch_size, 1, 32, 256, 256)
        sample = trainer.ema_model.module.sample(batch_size=batch_size, mask=mask)  # torch.Size([batch_size, 1, 32, 256, 256])
        
    for j in range(batch_size):
        np.save(save_path_list[j], sample.cpu().detach().numpy()[j][0])  # [32, 256, 256]