import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from multiprocessing import cpu_count


# Check for cuda availability
print(f'CUDA available: {torch.cuda.is_available()}')
print("---------------------------------------------------")


# Load env and login to huggingface
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_AUTH_TOKEN")
login(HUGGINGFACE_TOKEN)


def prepare_data():
    ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed")
    dataset_dict = {
            "train": ds["train"],
            "val": ds["val"]
    }
    return DatasetDict(dataset_dict)


def main():
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
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype="auto",
        use_cache=False,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    # Prevent the grad norm from getting too large
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)

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
        report_to="none",
        gradient_accumulation_steps=1
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