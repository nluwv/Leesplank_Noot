# Leesplank_Noot

This repository was created to develop the LLM Noot

# Preprocessing
It contains scripts for preprocessing the training data. There are 2 trainging datasets:
simplifictions:
    contains:
        - prompt: originional text
        - result: simplified output
    The processing script removes list like prompt, sorts the data on levenstein distance and saves the data on Huggingface
    https://huggingface.co/datasets/UWV/Leesplank_NL_wikipedia_simplifications_preprocessed 
veringewikkeldingen
    contains:
        - origional text
        - simplification
        - specific complex versions.
    The processing script reformats the dataset so that the simplification is the result and the complex versions are the prompt. 
    https://huggingface.co/datasets/UWV/veringewikkelderingen_preprocessed

In the server folder you find a markdown file containing some usefull commands to get the training started on the server and the documents needed on the server for training.

# Train.py
The script finetunes a Seq2Seq model for Dutch text simplification, specifically using the BramVanroy/ul2-small-dutch-simplification-mai-2023 model. It incorporates distributed training for scalability and performance optimization across multiple GPUs.

Key Techniques and Tools Used:
- Transformers Library: Uses Hugging Face's AutoTokenizer and AutoModelForSeq2SeqLM for tokenization and model loading.
- Distributed Training: Implemented with torch.distributed and DistributedDataParallel (DDP) to parallelize across devices.
- Data Loading: Uses datasets to load and process data, and DataLoader for batching.
- Optimization: AdamW optimizer with learning rate scheduling.
- Logging: Training logs are stored using Python's logging module.
Environment Variables: API keys (e.g., Hugging Face token) are loaded via .env.

Distributed Training:
https://www.youtube.com/watch?v=azLCUayJJoQ&t=591s
