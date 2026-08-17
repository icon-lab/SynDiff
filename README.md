# SynDiff

Official PyTorch implementation of SynDiff described in the [paper](https://ieeexplore.ieee.org/document/10167641).

Muzaffer Özbey*, Onat Dalmaz*, Salman UH Dar, Hasan A Bedel, Şaban Özturk, Alper Güngör, Tolga Çukur, "Unsupervised Medical Image Translation With Adversarial Diffusion Models," in IEEE Transactions on Medical Imaging, vol. 42, no. 12, pp. 3524-3539, Dec. 2023, doi: 10.1109/TMI.2023.3290149.

*: equal contribution

<img src="./figures/adv_diff.png" width="600px">

<img src="./figures/syndiff.png" width="600px">

## Dependencies

```
python>=3.6.9
torch>=1.13
torchvision>=0.14
numpy
h5py
scikit-image
```

`torch>=1.13` is required because the resume path passes `weights_only` to
`torch.load`; PyTorch 2.6 later flipped that argument's default to `True`,
which is why it is now passed explicitly.

### Optional: fused CUDA kernels
`utils/op` ships hand-written CUDA kernels that are JIT-compiled on first
import. Building them needs a CUDA toolchain and:

```
cuda>=11.2
ninja
python3.x-dev (apt install, x should match your python3 version, ex: 3.8)
```

If any of these is missing, SynDiff warns once and falls back to equivalent
pure-PyTorch implementations, so the code also runs on a CPU-only install.

## Installation
- Clone this repo:
```bash
git clone https://github.com/icon-lab/SynDiff
cd SynDiff
pip install -r requirements.txt
```

## Dataset
You should structure your aligned dataset in the following way:



```
input_path/
  ├── data_train_contrast1.mat
  ├── data_train_contrast2.mat
  ├── data_val_contrast1.mat
  ├── data_val_contrast2.mat
  ├── data_test_contrast1.mat
  ├── data_test_contrast2.mat
```

where the `contrast1`/`contrast2` parts of the file names are the values passed
to `--contrast1` and `--contrast2`.

Each `.mat` file is an HDF5 file holding a single variable named `data_fs` of
shape `(#images, width, height)`, with image values in roughly `[0, 1]`.
Volumes are zero-padded out to `--image_size` squared on load and rescaled to
`[-1, 1]`, so neither dimension may exceed that size.

### Sample Data
Sample toy data can be found under the `SynDiff_sample_data` folder. Note that
those two files are raw volumes (`T1.mat`, `T2.mat`, 25 slices each) rather
than a ready-made split -- to run the commands below, split them into train /
val / test parts and name the parts as shown above.



## Train

<br />

```
python3 train.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --contrast1 T1 --contrast2 T2 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 1 --save_content --local_rank 0 --input_path /input/path/for/data --output_path /output/for/results
```

`--ngf` sets the channel width of the discriminators *and* of the translation
networks, and `--image_size` sets both the network resolution and the grid the
input volumes are padded to. The values above, `64` and `256`, are the ones
used in the paper.

`--num_process_per_node` controls the number of processes. With more than one
a NCCL process group is set up and the networks are wrapped in
`DistributedDataParallel`; with a single process neither is used, and the run
falls back to CPU when no GPU is visible.

<br />

## Pretrained Models
We have released pretrained diffusive generators for [T1->PD and PD->T1](https://drive.google.com/file/d/1Hfvnz29NaTFqPMX6RGaEv4Qnt8HeoxZz/view?usp=sharing) tasks in IXI and [T1->T2 and T2->T1](https://drive.google.com/file/d/1zGzZPVY-Xp2Flc7GicOD7s4taxcjwCsn/view?usp=sharing) tasks in BRATS datasets. You can save these weights in relevant checkpoints folder and perform inference.

## Test

<br />

```
python test.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional  --z_emb_dim 256 --contrast1 T1  --contrast2 T2 --which_epoch 50 --gpu_chose 0 --input_path /input/path/for/data --output_path /output/for/results
```

Synthesised images are written to
`output_path/exp/generated_samples/epoch_<which_epoch>/`, both as JPEGs and
collected into `im_syn.mat`. Before saving, each image is cropped back from
the padded 256x256 grid; `--crop_h` and `--crop_w` set that size and default
to `256 152`, the slice geometry used in the paper. Set them to your own
slice size for other datasets.

<br />

## Tests

A CPU test suite covers the diffusion coefficients, network shapes and
gradients, dataset loading and checkpoint handling:

```
pip install -r requirements.txt
python -m pytest tests/
```

Tests that compare the fused CUDA kernels against their pure-PyTorch
fallbacks are skipped automatically when the extensions cannot be built.

<br />
<br />


# Citation
Preliminary versions of SynDiff are presented in [NeurIPS Medical Imaging Meets](https://www.cse.cuhk.edu.hk/~qdou/public/medneurips2022/105.pdf) and IEEE ISBI 2023.
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@ARTICLE{ozbey_dalmaz_syndiff_2024,
  author={Özbey, Muzaffer and Dalmaz, Onat and Dar, Salman U. H. and Bedel, Hasan A. and Özturk, Şaban and Güngör, Alper and Çukur, Tolga},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Unsupervised Medical Image Translation With Adversarial Diffusion Models}, 
  year={2023},
  volume={42},
  number={12},
  pages={3524-3539},
  keywords={Biological system modeling;Computational modeling;Training;Generative adversarial networks;Image synthesis;Task analysis;Generators;Medical image translation;synthesis;unsupervised;unpaired;adversarial;diffusion;generative},
  doi={10.1109/TMI.2023.3290149}}


```
For any questions, comments and contributions, please contact Muzaffer Özbey (muzafferozbey94[at]gmail.com) or Onat Dalmaz (onat[at]stanford.edu) <br />

(c) ICON Lab 2023

<br />

# Acknowledgements

This code uses libraries from, [pGAN](https://github.com/icon-lab/pGAN-cGAN), [StyleGAN-2](https://github.com/NVlabs/stylegan2), and [DD-GAN](https://github.com/NVlabs/denoising-diffusion-gan) repositories.
