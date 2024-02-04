#!/bin/bash 
#SBATCH -N 4
#SBATCH -p GPU-shared
#SBATCH -t 4:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --output=~/project/out/slurm-%j.out
set -x

# module load anaconda3
# cd $PROJECT
# conda activate envs/jh_clone
# cd mamba

python main.py