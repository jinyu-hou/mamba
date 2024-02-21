from datetime import datetime
from pytz import timezone
import time
from functools import partial
import wandb
import os
import fire
import tqdm
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from mamba_ssm.modules.mamba_simple import Mamba, Block
#from model_utils.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer

from main_utils import (
    load_jsonl_examples,
    get_cosine_lr_decay_fn,
    get_grad_norm,
    save_checkpoint,
    get_last_ckpt_idx, 
    create_dataloaders)


TIMEZONE = timezone('EST')
DATE = str(datetime.now(tz=TIMEZONE)).split()[0]
MODEL_SIZE = '3b'
PROJECT_NAME = f'mamba_{MODEL_SIZE}_lowrank'
RUN_NAME = f'ft_{MODEL_SIZE}_{DATE}'
HF_MODEL_NAME_OR_PATH = f'state-spaces/mamba-2.8b-slimpj'
WORKDIR = f'workdir_{MODEL_SIZE}'

LEARNING_RATE = 1e-4
# LR_SCHEDULE_TYPE = 'cosine'
END_LEARNING_RATE = 1e-5
# WARMUP_GRAD_STEPS = 2000
N_EPOCHS = 1
WARMUP_GRAD_STEPS = 2000
GRAD_NORM_CLIP = 1.
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
ACCELERATOR = 'cuda'
# PRECISION = 'bf16-mixed'
PRECISION = '32'
DTYPE = torch.float32
RANDOM_SEED = 11111

TRAIN_DATA_DIR = '/jet/home/jhou/project/huggingface/datasets/DKYoon___slim_pajama-6_b/prepared/train/'
VALID_DATA_DIR = '/jet/home/jhou/project/huggingface/datasets/DKYoon___slim_pajama-6_b/prepared/validation/'
#'/lustre/scratch/shared-folders/llm_project/refinedpajama_v1_llama_json_360s_shuffle/train'
TRAIN_EXAMPLES_PER_CHUNK = 6000000000  #1706976
N_CHUNKS = 4
BLOCK_SIZE = 2048

PRESERVE_RATE = 0.5
TOKENIZER_NAME = 'EleutherAI/gpt-neox-20b'


def collate_fn(examples, device):
    token_ids = torch.tensor(
        [example['token_ids'] for example in examples], device=device)
    return {'input_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}


def train_chunk(fabric,
                tokenizer,
                model,
                optimizer,
                lr_schedule_fn,
                train_dataloader,
                val_dataloader,
                per_device_batch_size,
                accumulate_grad_batches,
                # chunk_idx,
                run_wandb):
    # step = len(examples) // per_device_batch_size

    # example_batch_idxes = tqdm.trange(
    #     0, len(examples), per_device_batch_size,
    #     desc=f'Training (global_micro_batch_size='
    #          f'{per_device_batch_size * fabric.world_size}, '
    #          f'accumulate_grad_batches={accumulate_grad_batches})'
    # )
    
    for iter_num, train_data in enumerate(train_dataloader):
    # for i in example_batch_idxes:
        t0 = time.time()
        lr = lr_schedule_fn(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = ((iter_num + 1) % accumulate_grad_batches != 0)

        # batch = collate_fn(
        #     examples=examples[i:i+per_device_batch_size], device=fabric.device)
        # print(batch)
        input_ids = train_data[:, 0 : BLOCK_SIZE].contiguous()
        targets = train_data[:, 1 : BLOCK_SIZE + 1].contiguous() 
        
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            #logits = model(input_ids).logits
            logits = model(input_ids)[0]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape((-1, logits.size(-1))), targets.reshape(-1))

            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'loss': loss.item(),
            'learning_rate': lr,
            # 'step': step,
            'speed(#tok/s/gpu)': int(input_ids.numel() / (time.time() - t0))
        }
        if not is_accumulating:
            log['grad_norm'] = grad_norm

        example_batch_idxes.set_postfix(log)
        if run_wandb and fabric.global_rank == 0:
            wandb.log(log)

        break

    save_checkpoint(
        fabric=fabric,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        save_dir=f'{WORKDIR}/ckpt_test')


def main(n_nodes=1,
         n_devices_per_node=4,
         per_device_batch_size=4,
         accumulate_grad_batches=4,
         run_wandb=False):
    #torch.set_float32_matmul_precision('high')

    fabric = L.Fabric(
        accelerator=ACCELERATOR,
        num_nodes=n_nodes,
        devices=n_devices_per_node,
        precision=PRECISION,
        strategy=FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block}),
            activation_checkpointing_policy={Block},
            # sharding_strategy='HYBRID_SHARD',
            cpu_offload=True,
            limit_all_gathers=True))
    fabric.launch()

    if fabric.global_rank == 0:
        os.makedirs(WORKDIR, exist_ok=True)
        if run_wandb:
            wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    # last_ckpt_idx = get_last_ckpt_idx(workdir=WORKDIR)
    # fabric.seed_everything(RANDOM_SEED + last_ckpt_idx + 1)
    fabric.seed_everything(RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    model = MambaLMHeadModel.from_pretrained(HF_MODEL_NAME_OR_PATH, dtype=DTYPE, device=ACCELERATOR)
    # model.lowrank_decomp(preserve_rate=PRESERVE_RATE, device=ACCELERATOR, dtype=DTYPE)

    # model, optimizer = fabric.setup(model, optimizer)
    model = fabric.setup_module(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)

    # if last_ckpt_idx != -1:
    #     fabric.load(
    #         path=f'{WORKDIR}/ckpt_{last_ckpt_idx}/fabric_ckpt',
    #         state={'model': model, 'optimizer': optimizer})

    torch.cuda.empty_cache()

    # dataset = load_dataset(DATASET_NAME)
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=per_device_batch_size,
        block_size=BLOCK_SIZE,
        fabric=fabric,
        train_data_dir=TRAIN_DATA_DIR,
        val_data_dir=VALID_DATA_DIR,
        seed=1338,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    global_micro_batch_size = per_device_batch_size * fabric.world_size
    total_steps = TRAIN_EXAMPLES_PER_CHUNK // global_micro_batch_size
    lr_schedule_fn = get_cosine_lr_decay_fn(
        total_steps=total_steps,
        warmup_steps=WARMUP_GRAD_STEPS * accumulate_grad_batches,
        learning_rate=LEARNING_RATE,
        end_learning_rate=END_LEARNING_RATE)

    train_chunk(
        fabric=fabric,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        per_device_batch_size=per_device_batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        # chunk_idx=chunk_idx,
        run_wandb=run_wandb)

    # for chunk_idx in range(last_ckpt_idx + 1, N_CHUNKS):
    #     examples = load_jsonl_examples(
    #         filename=f'{TRAIN_DATA_DIR}/chunk_{chunk_idx}.jsonl',
    #         n_examples=TRAIN_EXAMPLES_PER_CHUNK,
    #         shuffle=True,
    #         global_micro_batch_size=global_micro_batch_size,
    #         global_rank=fabric.global_rank,
    #         world_size=fabric.world_size)

    #     train_chunk(
    #         fabric=fabric,
    #         tokenizer=tokenizer,
    #         model=model,
    #         optimizer=optimizer,
    #         lr_schedule_fn=lr_schedule_fn,
    #         examples=dataset["train"],
    #         per_device_batch_size=per_device_batch_size,
    #         accumulate_grad_batches=accumulate_grad_batches,
    #         chunk_idx=chunk_idx,
    #         run_wandb=run_wandb)


if __name__ == '__main__':
    fire.Fire(main)