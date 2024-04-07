#!/bin/bash
#SBATCH --job-name="mamba-lrft"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --account=bcmr-delta-gpu
#SBATCH --gpu-bind=closest
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 2-00:00:00
set -x

# cd $SCRATCH
# module load anaconda3_gpu
# conda init bash
# conda activate envs/fiona
# source activate $SCRATCH/envs/fiona
# cd mamba/finetuning

accelerate launch --config_file $SCRATCH/huggingface/accelerate/mamba_lrft.yaml hf_finetuning.py
# accelerate launch hf_finetuning.py
# python -m torch.distributed.launch hf_finetuning.py