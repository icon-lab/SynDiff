#!/bin/bash

#SBATCH --job-name=Diffusion

#resources:

#SBATCH --cpus-per-task=4

#SBATCH --partition=student

#SBATCH --mem-per-cpu=3G

#SBATCH --gpus=1

#SBATCH --time=48:00:00

# the maximum time the scripts needs to run

# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=slurm/outputs/errors/job.%J.err

#SBATCH --output=slurm/outputs/testjob%J.out

#python train.py --image_size 256 --exp exp_syndiff_T1fl2d --beta_min 0.02 --use_ema --ema_decay 0.999 --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --num_epoch 50 --ngf 64 --embedding_type positional  --r1_gamma 5 --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.5e-4 --lazy_reg 10 --num_process_per_node 1 --save_content --local_rank 0 --contrast1 T1_mapping_fl2d --contrast2 DIXON_T1_mapping_fl2d --input_path /home/students/studweilc1/SynDiff/my_data_group --output_path /home/students/studweilc1/SynDiff/my_results_group

# beta1=0.5, lr_D = 1e-4

export CUDA_HOME=/home/students/studweilc1/.conda/envs/cornelius_new
python train.py --image_size 256 --exp exp_syndiff_T1fl2d_registered --lr_g 1.0e-4 --lr_d 0.1e-4 --beta_min 0.02 --use_ema --ema_decay 0.999 --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --num_epoch 50 --ngf 64 --embedding_type positional  --r1_gamma 5 --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.5e-4 --lazy_reg 10 --num_process_per_node 1 --save_content --local_rank 0 --contrast1 T1_mapping_fl2d --contrast2 DIXON_T1_mapping_fl2d --input_path /home/students/studweilc1/SynDiff/data/my_data_group_registered --output_path /home/students/studweilc1/SynDiff/results/my_results_group_registered

#python train.py --image_size 256 --exp brats_exp --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --num_epoch 20 --ngf 64 --embedding_type positional  --r1_gamma 5 --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 1 --save_content --local_rank 0 --contrast1 T1 --contrast2 T2 --input_path /home/students/studweilc1/SynDiff/brats_data --output_path /home/students/studweilc1/SynDiff/my_results_brats

echo DONE!