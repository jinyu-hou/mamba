#!/bin/bash
#SBATCH --job-name="mamba-eval"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=2   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcmr-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 04:00:00
set -x

# module load anaconda3
# conda init bash
# cd $PROJECTS
# conda activate envs/fiona
# cd mamba

timestamp=$(date +%s)
# n_gpus=$(nvidia-smi --query-gpu=count --format=csv | tail -1)
result_dir=evals/results
model_id=mamba-1.4b
# model_id=mamba-2.8b
# model_id=mamba-2.8b-slimpj
result_prefix=$(echo $model_id | tr - _)
preserve_rate=0.5
preserve_percentage=$(awk -vn=$preserve_rate 'BEGIN{printf("%.0f\n",n*100)}')

# accelerate launch evals/lm_harness_eval.py \
#     --model mamba \
#     --model_args pretrained=state-spaces/$model_id,preserve_rate=$preserve_rate \
#     --device cuda \
#     --tasks arc_easy \
#     --batch_size 64 \
#     --output_path ${result_dir}/$timestamp.jsonl

rm ${result_dir}/${result_prefix}-$preserve_percentage.jsonl 
accelerate launch --config_file $SCRATCH/huggingface/accelerate/mamba_eval.yaml evals/lm_harness_eval.py \
    --model mamba \
    --model_args pretrained=state-spaces/$model_id,preserve_rate=$preserve_rate \
    --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,race,truthfulqa_mc2 \
    --device cuda \
    --batch_size 4 \
    --output_path ${result_dir}/${result_prefix}-$preserve_percentage.jsonl 

# rm ${result_dir}/${result_prefix}-$preserve_percentage-mmlu.jsonl
# accelerate launch evals/lm_harness_eval.py \
#     --model mamba \
#     --model_args pretrained=state-spaces/$model_id,preserve_rate=$preserve_rate \
#     --tasks mmlu \
#     --num_fewshot 5 \
#     --device cuda \
#     --batch_size 40 \
#     --output_path ${result_dir}/${result_prefix}-$preserve_percentage-mmlu.jsonl

# CUDA_VISIBLE_DEVICES=1 accelerate launch evals/lm_harness_eval.py \
#     --model mamba \
#     --model_args pretrained=state-spaces/$model_id,preserve_rate=$preserve_rate \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,race,truthfulqa_mc2 \
#     --device cuda \
#     --batch_size 32 \
#     --output_path ${result_dir}/${result_prefix}_A-$preserve_percentage.jsonl 

# python evals/results_to_csv.py
