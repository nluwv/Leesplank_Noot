import os
import torch
import datetime
from peft import LoraConfig
import torch.distributed as dist
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from datasets import DatasetDict
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print("---------------------------------------------------")

# Load env and login to huggingface
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_AUTH_TOKEN")
login(HUGGINGFACE_TOKEN)


def setup_distributed():
    """Initialize distributed training with correct GPU assignment"""
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ensure each process gets a unique GPU
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    print(f"Rank {rank} using GPU {local_rank} (total GPUs: {torch.cuda.device_count()})")


def prepare_data():
    ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed")
    dataset_dict = {
            "train": ds["train"],
            "val": ds["val"]
    }
    return DatasetDict(dataset_dict)


def main():
    setup_distributed()
    
    # Check if GPU assignment is correct
    print(f"Process {dist.get_rank()} is using GPU {torch.cuda.current_device()}")

    # Load the data
    ds = prepare_data()

    # Get tokenizer
    model_id = "ibm-granite/granite-3.1-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"<|system|>\nJij bent een expert in het vereenvoudigen van nederlandse teksten.\n<|user|>\n{example['instruction'][i]} {example['prompt'][i]}\n<|assistant|>\n{example['result'][i]}<|endoftext|>"
            output_texts.append(text)
        return output_texts
    response_template = "\n<|assistant|>\n"

    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype="auto",
        use_cache=False,
        device_map={"": torch.cuda.current_device()},
        quantization_config=quantization_config,
    )

    # Prevent the grad norm from getting too large (nan)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

    # Apply qLoRA
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )

    output_dir = "qlora_output/ibm3-1-2b-base-sft-lora"
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        max_steps=-1,
        do_eval=True,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        gradient_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        ddp_backend='nccl'
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['val'],
        processing_class=tokenizer,
        peft_config = peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    train_result = trainer.train()

    metrics= train_result.metrics
    max_train_samples = len(ds['train'])
    metrics["train_samples"] = min(max_train_samples, len(ds['val']))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()