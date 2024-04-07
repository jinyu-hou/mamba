import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments
from mamba_ssm.models.mixer_seq_lowrank import MambaForCausalLM
# from mamba_ssm.models.mixer_seq_simple import MambaForCausalLM

# model_id = "state-spaces/mamba-130m"
model_id = "state-spaces/mamba-2.8b-slimpj"
dataset_id = "DKYoon/SlimPajama-6B"
tokenizer_id = "EleutherAI/gpt-neox-20b"
dtype=torch.bfloat16
device="cuda"
preserve_rate=0.25

model = MambaForCausalLM.from_pretrained(model_id, dtype=dtype, device=device)
model.lowrank_decomp(preserve_rate=float(preserve_rate))

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset_train = load_dataset(dataset_id, split="train[:50%]")
dataset_val = load_dataset(dataset_id, split="validation")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    evaluation_strategy="steps",
    save_steps=100,
    optim_target_modules=["in_proj_A", "in_proj_B", "out_proj_A", "out_proj_B"],
    load_best_model_at_end=True,
    save_total_limit=2,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    warmup_steps=2000*4
    # optim_target_modules=["in_proj", "out_proj"]
)
# lora_config =  LoraConfig(
#         r=8,
#         target_modules=["in_proj_A", "in_proj_B", "out_proj_A", "out_proj_B"],
#         task_type="CAUSAL_LM",
#         bias="none"
# )
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    # peft_config=lora_config,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    dataset_text_field="text",
)
trainer.train()