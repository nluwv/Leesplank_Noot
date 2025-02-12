import os
import torch
from peft import LoraConfig
import torch.distributed as dist
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from datasets import DatasetDict
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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

    # Get tokenizer
    model_id = "ibm-granite/granite-3.1-2b-instruct"
    processing_class = AutoTokenizer.from_pretrained(model_id)
    processing_class.padding_side = 'right'

    if processing_class.pad_token_id is None:
        processing_class.pad_token_id = processing_class.eos_token_id

    if processing_class.model_max_length > 100_000:
        processing_class.model_max_length = 2048
        
    # Load the data
    ds = prepare_data(processing_class)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"<|system|>\nJij bent een expert in het vereenvoudigen van nederlandse teksten.\n<|user|>\n{example['instruction'][i]} {example['prompt'][i]}\n<|assistant|>\n{example['result'][i]}<|endoftext|>"
            output_texts.append(text)
        return output_texts
    response_template = "\n<|assistant|>\n"

    response_template_ids = processing_class.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=processing_class)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
        torch_dtype="auto",
        use_cache=False,
        device_map={"": torch.cuda.current_device()},
        quantization_config=quantization_config,
    )

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
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        max_steps=-1,
        do_eval=True,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        logging_steps=50,
        bf16=True,  # try this instead of fp16
        optim="adafactor",
        gradient_checkpointing=True,
        max_grad_norm=5,  # Prevent grad norm from getting too large
        report_to="none",
        gradient_accumulation_steps=2,
        ddp_find_unused_parameters=False,
        ddp_backend='nccl'
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['val'],
        processing_class=processing_class,
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