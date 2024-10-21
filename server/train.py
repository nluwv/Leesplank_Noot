import os
from dotenv import load_dotenv
import glob
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import multiprocessing
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from datasets import load_dataset
import sentencepiece
import accelerate
from huggingface_hub import login
import evaluate
from transformers import get_scheduler
from tqdm import tqdm
import logging

#set 
logging.basicConfig(filename='training.log', level=logging.INFO)

# Load environment variables (API keys) from .env file
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize the distributed process group
dist.init_process_group(backend='gloo')

# Set device based on local rank
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(local_rank)

# Load your model and send it to the correct device
tokenizer = AutoTokenizer.from_pretrained("BramVanroy/ul2-small-dutch-simplification-mai-2023", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("BramVanroy/ul2-small-dutch-simplification-mai-2023").to(device)

# Wrap model in DDP
if torch.cuda.device_count() > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)


# Download dataset
ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split="train") 


print('Data loaded')

# Tokenization function with padding and truncation
def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["result"], padding="max_length", truncation=True)

# Tokenize datasets
tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=['prompt', 'result', '__index_level_0__'])
print('Data is tokenized')

# Split the tokenized dataset (e.g., 80% train, 20% validation)
splits = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = splits["train"]
valid_dataset = splits["test"]
print('Train-validation split is done')


# Set up DistributedSampler for training and validation datasets
train_sampler = DistributedSampler(train_dataset)
valid_sampler = DistributedSampler(valid_dataset)

# Data collator for Seq2Seq task, with padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, collate_fn=data_collator)
valid_loader = DataLoader(valid_dataset, batch_size=8, sampler=valid_sampler, collate_fn=data_collator)

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
    save_strategy="steps",  
    save_steps=1000,  
    logging_dir="./logs",  
    logging_steps=100,
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available()  # Enable mixed precision training on ROCm-enabled AMD GPUs
)

print('Training arguments set')

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
print('Optimizer initialized')

lr_scheduler = get_scheduler(
    name="linear",  # Type of scheduler: linear, cosine, etc.
    optimizer=optimizer,
    num_warmup_steps=0,  # You can set this based on your needs
    num_training_steps=len(train_loader) * training_args.num_train_epochs
)


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
print('Trainer initialized')
logging.info('Trainer initialized')

# Train the model
#trainer.train()

# Train the model
print('Start training')
for epoch in range(training_args.num_train_epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(train_loader):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        # Log loss to progress bar
        progress_bar.set_postfix(loss=loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
        # Save model every 1000 steps
        if step % 1000 == 0 and step > 0:  # Ensure we don't save at step 0
            print(f"Saving model at step {step}...")
            trainer.save_model('./results')
            logging.info(f"Model saved at step {step}")

        # Update learning rate (if using a scheduler)
        if trainer.lr_scheduler is not None:
            last_lr = trainer.lr_scheduler.get_last_lr()[0]
        else:
            last_lr = training_args.learning_rate

        # Log
        if step % 100 == 0:
            logging.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
            logging.info(f"Learning rate: {last_lr}")

    # Update progress bar
    progress_bar.update(1)

print('Training has finished')
logging.info('Training has finished')



print('Training has finished')
logging.info('Training has finished')

# Save model
logging.info('Start saving model')
print('Start saving model')
# local:
model.save_pretrained("./saved_model")

login(HUGGINGFACE_TOKEN)
trainer.push_to_hub("UWV/ul2-small-dutch-simplification-okt-2024")
print('Model saved')
logging.info('Model saved')

