import polars as pl
import torch
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
HUGGINFACE_TOKEN = os.getenv("HUGGINFACE_TOKEN")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("BramVanroy/ul2-small-dutch-simplification-mai-2023", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("BramVanroy/ul2-small-dutch-simplification-mai-2023")

df = pl.read_parquet('hf://datasets/UWV/Leesplank_NL_wikipedia_simplifications_preprocessed/data/train-*.parquet')
print(df.columns)
ds = Dataset.from_pandas(df.to_pandas())

# Check if you have a GPU, otherwise default to CPU
if torch.backends.mps.is_available():  # Check for AMD ROCm GPU
    device = torch.device("mps")
    print("mps")
else:
    device = torch.device("cpu")
    print("cpu")
model.to(device)


def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["result"], padding="max_length", truncation=True)

# Tokenize datasets
tokenized_datasets = ds.map(tokenize_function, batched=True)

# Split the tokenized dataset (e.g., 80% train, 20% validation)
splits = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = splits["train"]
valid_dataset = splits["test"]



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


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
    output_dir="./results",
    evaluation_strategy="epoch",
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

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save model
login('HUGGINGFACE_TOKEN')
trainer.push_to_hub("UWV/ul2-small-dutch-simplification-okt-2024")