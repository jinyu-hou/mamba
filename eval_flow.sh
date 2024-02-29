#!/bin/bash 
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 4:00:00
#SBATCH --gpus=v100-32:4
#SBATCH -o ~/project/out/slurm-%j.out
set -x

# module load anaconda3
# cd $PROJECT
# conda activate envs/jh_clone
# cd mamba

timestamp=$(date +%s)
# n_gpus=$(nvidia-smi --query-gpu=count --format=csv | tail -1)
result_dir=evals/results
model_id=mamba-1.4b
# model_id=mamba-2.8b
result_prefix=$(echo $model_id | tr - _)
preserve_rate=1.0
preserve_percentage=$(awk -vn=$preserve_rate 'BEGIN{printf("%.0f\n",n*100)}')

accelerate launch evals/lm_harness_eval.py \
    --model mamba \
    --model_args pretrained=state-spaces/$model_id,preserve_rate=$preserve_rate \
    --device cuda \
    --tasks arc_easy \
    --batch_size 64 \
    --output_path ${result_dir}/$timestamp.jsonl

# rm ${result_dir}/${result_prefix}-$preserve_percentage.jsonl 
# accelerate launch evals/lm_harness_eval.py \
#     --model mamba \
#     --model_args pretrained=state-spaces/$model_id,preserve_rate=$preserve_rate \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,race,truthfulqa_mc2 \
#     --device cuda \
#     --batch_size 40 \
#     --output_path ${result_dir}/${result_prefix}-$preserve_percentage.jsonl 

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
