import torch

import transformers
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_lowrank import MambaLMHeadModel

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from os.path import dirname
from pathlib import Path
import json

@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="state-spaces/mamba-2.8b", max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.float16, preserve_rate=1.0):
        LM.__init__(self)
        self._model = MambaLMHeadModel.from_pretrained(pretrained, dtype=dtype, device=device)
        self._model.lowrank_decomp(preserve_rate=float(preserve_rate), device=device, dtype=dtype)
        self.export_el_count(pretrained, float(preserve_rate))
        if device == "cuda":
            self._model = self.init_ddp(self._model)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    def init_ddp(self, model):
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running with basic DDP on rank {rank}.")

        # create model and move it to GPU with id rank
        device_id = rank % torch.cuda.device_count()
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])
        print(f"Device ID: {device_id}")
        return ddp_model   

    def export_el_count(self, pretrained, preserve_rate):
        model_id = pretrained.split("/")[1].replace("-", "_")
        outfile = "{}/results/{}-{}-numel.jsonl".format(
            dirname(__file__), model_id, int(preserve_rate*100))
        if Path(outfile).is_file():
            return 
        el_count = 0
        for _, param in self._model.state_dict().items():
            el_count += torch.numel(param)
        with open(outfile, "w") as f:
            f.write(json.dumps({"numel": el_count}, indent=2))

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
    dist.destroy_process_group()
