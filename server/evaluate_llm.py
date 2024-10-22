import random
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import evaluate

# Load the model and tokenizer
model_path = "./results"  # Update this path
model = AutoModelForSeq2SeqLM.from_pretrained(model_path) 
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the dataset
ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split="train")

# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Function to compute metrics
def compute_metrics(predictions, labels):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU and ROUGE
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu['bleu'],
        "rouge1": rouge['rouge1'],
        "rouge2": rouge['rouge2']
    }

# Select 100 random cases from the dataset
random_samples = random.sample(range(len(ds)), 100)
inputs = []
labels = []
outputs = []

for idx in random_samples:
    input_text = ds[idx]['prompt']  
    label_text = ds[idx]['result']   

    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # Generate output
    with torch.no_grad():  # Disable gradient calculation
        output_ids = model.generate(input_ids, max_length=50)  # Adjust max_length as needed
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Store input, label, and output
    inputs.append(input_text)
    labels.append(label_text)
    outputs.append(output_text)

# Convert labels to tensor for metrics computation
labels_tensor = tokenizer(labels, padding=True, truncation=True, return_tensors='pt').input_ids

# Compute metrics
predictions_tensor = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt').input_ids
metrics = compute_metrics(predictions_tensor, labels_tensor)

# Create a DataFrame for results
results_text_df = pd.DataFrame({
    'input': inputs,
    'label': labels,
    'output': outputs
})

results_metrics_df = pd.DataFrame({
    'bleu': [metrics['bleu']],  
    'rouge1': [metrics['rouge1']],  
    'rouge2': [metrics['rouge2']]
})

# Save to CSV
results_text_df.to_csv('model_output.csv', index=False)
results_metrics_df.to_csv('model_output_with_metrics.csv', index=False)
