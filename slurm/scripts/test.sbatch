#!/bin/bash

#SBATCH --job-name=Diffusion

#resources:

#SBATCH --cpus-per-task=4

#SBATCH --partition=student

#SBATCH --mem-per-cpu=3G

#SBATCH --gpus=1

#SBATCH --time=16:00:00

# the maximum time the scripts needs to run

# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=slurm/outputs/errors/job.%J.err

#SBATCH --output=slurm/outputs/testjob%J.out
#!/bin/bash

#SBATCH --job-name=Diffusion

#resources:

#SBATCH --cpus-per-task=4

#SBATCH --partition=student

#SBATCH --mem-per-cpu=3G

#SBATCH --gpus=1

#SBATCH --time=16:00:00

# the maximum time the scripts needs to run

# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=slurm/outputs/errors/job.%J.err

#SBATCH --output=slurm/outputs/testjob%J.out

python test.py --image_size 256 --exp exp_syndiff2 --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional  --z_emb_dim 256 --contrast1 T1  --contrast2 DIXON --which_epoch 10 --gpu_chose 0 --input_path /home/students/studweilc1/SynDiff/my_data --output_path /home/students/studweilc1/SynDiff/my_results


echo DONE!