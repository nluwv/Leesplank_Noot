import os
from dotenv import load_dotenv
import glob
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
import multiprocessing
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from datasets import load_dataset, Dataset, DatasetDict
import sentencepiece
import accelerate
from huggingface_hub import login
import evaluate

# Load environment variables (API keys) from .env file
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("BramVanroy/ul2-small-dutch-simplification-mai-2023", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("BramVanroy/ul2-small-dutch-simplification-mai-2023")

print('model loaded')




# Check if you have a GPU, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Detect and use multiple GPUs (AMD ROCm)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = torch.nn.parallel.DistributedDataParallel(model)

# Use multiprocessing to utilize all CPU cores if no GPU is available
if device == "cpu":
    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPU cores for training.")





# Download dataset
ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split="train") 
print('Data loaded')


# Tokenization function with padding and truncation
def tokenize_function(examples):
    # Tokenizing the 'prompt' and 'result' with padding and truncation
    return tokenizer(examples["prompt"], text_target=examples["result"], padding="max_length", truncation=True)

# Tokenize datasets
tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=['prompt', 'result', '__index_level_0__'])
print('Data is tokenized')


# Split the tokenized dataset (e.g., 80% train, 20% validation)
splits = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = splits["train"]
valid_dataset = splits["test"]

print(train_dataset[0])


print('Train-validation split is done')

# Data collator for Seq2Seq task, with padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")



# Load BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Define a function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU and ROUGE
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "bleu": bleu['bleu'],
        "rouge": rouge['rougeL'].mid.fmeasure
    }

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    remove_unused_columns=False,
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Enable mixed precision training on ROCm-enabled AMD GPUs
    save_strategy="epoch"
)

print('training arguments set')

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
print('optimizer initialised')

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)
print('trainer initialized')

# Train the model
print('start training')
trainer.train()
print('training has finished')


# Save model
print('start saving model')
login('HUGGINGFACE_TOKEN')
trainer.push_to_hub("UWV/ul2-small-dutch-simplification-okt-2024")
print('done saving model')