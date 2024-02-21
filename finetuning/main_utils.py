import os
import glob
import json
import tqdm
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from lightning.fabric.strategies import FSDPStrategy

from mamba_ssm.models.mixer_seq_lowrank import MambaLMHeadModel
#from model_utils.modeling_llama import LlamaForCausalLM

from packed_dataset import PackedDataset


def load_jsonl_examples(filename,
                        n_examples,
                        shuffle,
                        global_micro_batch_size,
                        global_rank,
                        world_size):
    example_idxes = np.random.permutation(n_examples) if shuffle \
        else np.arange(n_examples)

    n_examples = n_examples // global_micro_batch_size * global_micro_batch_size
    example_idxes = example_idxes[global_rank:n_examples:world_size]

    examples = {idx: None for idx in example_idxes}
    for example_idx, line in tqdm.tqdm(
            enumerate(open(filename)), desc=f'loading {filename}'):
        if example_idx in examples:
            examples[example_idx] = json.loads(line)

    return [examples[idx] for idx in example_idxes]


def get_cosine_lr_decay_fn(total_steps,
                           warmup_steps,
                           learning_rate,
                           end_learning_rate):
    def cosine_with_warmup_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        elif step > total_steps:
            return end_learning_rate

        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return end_learning_rate + coeff * (learning_rate - end_learning_rate)

    return cosine_with_warmup_lr


def get_grad_norm(model):
    square_sum = 0.
    for param in model.parameters():
        if param.grad is not None:
            square_sum += param.grad.detach().data.norm(2).item() ** 2
    return square_sum ** 0.5


def save_checkpoint(fabric, tokenizer, model, optimizer, save_dir):
    assert isinstance(fabric.strategy, FSDPStrategy)

    save_policy = FullStateDictConfig(
        offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
    with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=save_policy):
        state_dict = model._forward_module.state_dict()

    if fabric.global_rank == 0:
        tokenizer.save_pretrained(save_dir)
        assert isinstance(model, MambaLMHeadModel)
        # model.module.save_pretrained(
        #     save_dir, state_dict=state_dict, safe_serialization=False)

        torch.save(state_dict, f'{save_dir}/pytorch_model.bin')
        json.dump(
            model.module.config.__dict__, open(f'{save_dir}/config.json', 'w'))

    fabric.barrier()
    fabric.save(
        path=f'{save_dir}/fabric_ckpt',
        state={'model': model, 'optimizer': optimizer})


def get_last_ckpt_idx(workdir):
    last_ckpt_idx = -1
    for ckpt_dir in glob.glob(f'{workdir}/ckpt_*'):
        ckpt_idx = int(ckpt_dir.split('_')[-1])
        if ckpt_idx > last_ckpt_idx:
            last_ckpt_idx = ckpt_idx

    return last_ckpt_idx


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: str,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
) -> DataLoader:
    datasets = []
    filenames = glob.glob(os.path.join(data_dir + "*"))
    dataset = PackedDataset(
        filenames, n_chunks=4, block_size=block_size, shuffle=shuffle, seed=seed,
        num_processes=fabric.world_size, process_rank=fabric.global_rank,
    )

    if not dataset:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: str = "/jet/home/jhou/project/huggingface/datasets/DKYoon___slim_pajama-6_b/prepared/train",
    val_data_dir = "/jet/home/jhou/project/huggingface/datasets/DKYoon___slim_pajama-6_b/prepared/validation",
    seed: int = 12345,
):
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader