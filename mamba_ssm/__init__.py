__version__ = "1.1.1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_lowrank import Mamba
from mamba_ssm.models.mixer_seq_lowrank import MambaForCausalLM
# from mamba_ssm.models.mixer_seq_simple import MambaForCausalLM
