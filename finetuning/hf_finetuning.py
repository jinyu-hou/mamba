import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments
from mamba_ssm.models.mixer_seq_lowrank import MambaLMHeadModel

model_id = "state-spaces/mamba-130m"
dtype=torch.float32
device="cuda"
preserve_rate=0.25

model = MambaLMHeadModel.from_pretrained(model_id, dtype=dtype, device=device)
model.lowrank_decomp(preserve_rate=float(preserve_rate))

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
dataset = load_dataset("Abirate/english_quotes", split="train")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        # target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        target_modules=["in_proj_lowrank", "out_proj_lowrank"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)
trainer.train()