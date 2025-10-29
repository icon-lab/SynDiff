import sys, os, types, math, pathlib, warnings
from types import SimpleNamespace

path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(1, path_to_pip_installs)

import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import CenterCrop
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dataset import CreateDatasetSynthesis
from backbones.ncsnpp_generator_adagn import NCSNpp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_values(input_path, contrast1, contrast2, ckpt_1, ckpt_2, split="test", max_samples=None):

    dataset_val = CreateDatasetSynthesis(
        phase=split,
        input_path=input_path,
        contrast1=contrast1,
        contrast2=contrast2
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_val,
        num_replicas=1,
        rank=0
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )

    # ---- import the exact functions/classes used in training ----
    from train import Posterior_Coefficients, get_time_schedule, sample_from_model
    from backbones.ncsnpp_generator_adagn import NCSNpp

    # ---- minimal args namespace matching training defaults the model expects ----
    from types import SimpleNamespace
    args = SimpleNamespace(
        # diffusion / sampling
        num_timesteps=4,
        nz=100,

        # NCSNpp architecture args (use the same values you trained with;
        # these mirror train.py defaults)
        image_size=256,
        num_channels=2,
        num_channels_dae=64,
        n_mlp=3,
        ch_mult=[1, 1, 2, 2, 4, 4], 
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.,
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=[1,3,3,1],
        skip_rescale=True,
        resblock_type='biggan',
        progressive='none',
        progressive_input='residual',
        progressive_combine='sum',
        embedding_type='positional',
        fourier_scale=16.,
        not_use_tanh=False,
        z_emb_dim=256,
        t_emb_dim=256,
        beta_min=0.02,
        ngf=64,
        r1_gamma=5,
        num_process_per_node=1,
        lazy_reg=10,
        beta_max=20,
        use_geometric=False,
        centered=True,
    )

    # ---- build generators (one per direction) and load checkpoints ----
    gen1 = NCSNpp(args).to(device).eval()
    gen2 = NCSNpp(args).to(device).eval()

    def load_stripping_module(model, ckpt_path):
        """Load a state_dict possibly saved from DDP ('module.' prefix)."""
        sd = torch.load(ckpt_path, map_location=device)
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)

    load_stripping_module(gen1, ckpt_1)
    load_stripping_module(gen2, ckpt_2)

    # ---- build coeffs and time schedule exactly like training ----
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # ---- evaluate PSNR the same way as training ----
    psnrs_dir1 = []  
    psnrs_dir2 = []
    ssim_dir1 = []
    ssim_dir2 = []

    # Direction 1: (x_val, y_val) => use gen1
    with torch.no_grad():
        for i,(x_val, y_val) in enumerate(data_loader_val):
            real_data = x_val.to(device, non_blocking=True)    # contrast1 (target)
            source_data = y_val.to(device, non_blocking=True)  # contrast2 (source)

            # concatenate noise + source (exactly like training)
            x1_t = torch.cat((torch.randn_like(real_data), source_data), dim=1)

            # sample from model (same call / args as training)
            fake = sample_from_model(pos_coeff, gen1, args.num_timesteps, x1_t, T, args)
            fake_np = to_range_0_1(fake).cpu().numpy()
            fake_np = fake_np/fake_np.mean()
            real_np = to_range_0_1(real_data).cpu().numpy()
            real_np = real_np/real_np.mean()

            p = psnr(real_np, fake_np, data_range=real_np.max())
            psnrs_dir1.append(p)

            s = ssim(fake_np[0,0,:,:], real_np[0,0,:,:], data_range=real_np.max())
            ssim_dir1.append(s)

            if max_samples is not None and i >= max_samples:
                break

    # Direction 2: (y_val, x_val) => use gen2
    with torch.no_grad():
        for i,(y_val, x_val) in enumerate(data_loader_val):
                
            real_data = x_val.to(device, non_blocking=True)    # contrast2->contrast1 direction target
            source_data = y_val.to(device, non_blocking=True)  # source

            x1_t = torch.cat((torch.randn_like(real_data), source_data), dim=1)
            fake = sample_from_model(pos_coeff, gen2, args.num_timesteps, x1_t, T, args)

            fake_np = to_range_0_1(fake).cpu().numpy()
            fake_np = fake_np/fake_np.mean()
            real_np = to_range_0_1(real_data).cpu().numpy()
            real_np = real_np/real_np.mean()
            p = psnr(real_np, fake_np, data_range=real_np.max())
            psnrs_dir2.append(p)

            s = ssim(fake_np[0,0,:,:], real_np[0,0,:,:], data_range=real_np.max())
            ssim_dir2.append(s)
            
            if max_samples is not None and i >= max_samples:
                break


    print("Validation PSNR Values:")
    print("Class 1:")
    print(np.nanmean(psnrs_dir1))
    print("Class 2:")
    print(np.nanmean(psnrs_dir2))
    print("Class 1 SSIM:")
    print(np.nanmean(ssim_dir1))
    print("Class 2 SSIM:")
    print(np.nanmean(ssim_dir2))
    return np.nanmean(psnrs_dir1), np.nanmean(psnrs_dir2), np.nanmean(ssim_dir1), np.nanmean(ssim_dir2)

def evaluate_all_cases():
    input_path = "data/my_data_group"
    contrast1 = "T1_mapping_fl2d"
    contrast2 = "DIXON"

    cases = {
        "T1_DIXON_test": {
            "ckpt_1": "checkpoints/case1_gen1.pth",
            "ckpt_2": "checkpoints/case1_gen2.pth",
            "contrast1": "T1_mapping_fl2d",
            "contrast2": "DIXON",
            "split": "test"
        },
        "T1_DIXON_val": {
            "ckpt_1": "checkpoints/case1_gen1.pth",
            "ckpt_2": "checkpoints/case1_gen2.pth",
            "contrast1": "T1_mapping_fl2d",
            "contrast2": "DIXON",
            "split": "val"
        },
        "T1_DIXON_train": {
            "ckpt_1": "checkpoints/case1_gen1.pth",
            "ckpt_2": "checkpoints/case1_gen2.pth",
            "contrast1": "T1_mapping_fl2d",
            "contrast2": "DIXON",
            "split": "train"
        },

    }



if __name__ == "__main__":
    evaluate_all_cases()

