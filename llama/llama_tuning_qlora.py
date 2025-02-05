import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from multiprocessing import cpu_count

print(f'CUDA available: {torch.cuda.is_available()}')
print("---------------------------------------------------")

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_AUTH_TOKEN")

login(HUGGINGFACE_TOKEN)

ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed")

dataset_dict = {
        "train": ds["train"],
        "val": ds["val"]
}
ds = DatasetDict(dataset_dict)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048

def apply_chat_template(row):
    messages = [
        {"role": "user", "content": f"""{row['instruction']} {row['prompt']}"""},
        {"role": "assistant", "content": row["result"]}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt = True
    )

    return {"text": prompt}

column_names = list(ds["train"].features)
ds = ds.map(
        apply_chat_template,
        num_proc=cpu_count(),
#        fn_kwargs = {"tokenizer": tokenizer},
        remove_columns=column_names,
        desc="Applying chat template",
)

ds_train = ds["train"]
ds_val = ds["val"]

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="eager",
    torch_dtype="auto",
    use_cache=False,
    device_map=device_map,
    quantization_config=quantization_config,
)

output_dir = "qlora_output/llama3.2-3b-sft-lora"
training_args = SFTConfig(
    fp16=True,
    do_eval=True,
    eval_strategy="epoch",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    save_strategy="no",
    save_total_limit=None,
    seed=42,
    dataset_text_field="text",
    max_seq_length=tokenizer.model_max_length,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

train_result = trainer.train()

metrics= train_result.metrics
max_train_samples = len(ds_train)
metrics["train_samples"] = min(max_train_samples, len(ds_train))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()