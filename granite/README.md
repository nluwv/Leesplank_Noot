# Granite Dutch Text Simplification

Fine-tuning IBM Granite 3.3 2B Instruct model for Dutch text simplification using the Leesplank dataset.

## Overview

This module provides a complete pipeline for fine-tuning the `ibm-granite/granite-3.3-2b-instruct` model to simplify Dutch texts to a clear, accessible level for everyone. The model uses language similar to how Jeugdjournaal presents information - clear and accessible without being childish. It's trained on the `UWV/Leesplank_NL_wikipedia_simplifications_preprocessed` dataset, which contains Wikipedia articles and their simplified versions.

## Features

- **Flexible Training Options**: Support for both full fine-tuning and LoRA/QLoRA
- **Granite Chat Template**: Proper formatting for Granite's conversation structure  
- **Interactive & Batch Inference**: Test simplification modes for different use cases
- **Modular Design**: Clean, configurable, and maintainable codebase
- **Memory Efficient**: Optimized for training on consumer GPUs

## Repository Structure

```
granite/
├── configs/
│   ├── training_config.yaml      # Training hyperparameters and settings
│   └── model_config.yaml         # Model and LoRA/quantization configuration
├── src/
│   ├── data_processing.py        # Dataset loading and chat template formatting
│   └── model_utils.py            # Model loading with quantization and LoRA
├── scripts/
│   ├── train.py                  # Main training script
│   └── inference.py              # Inference script (interactive/batch)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU with 16GB+ VRAM (recommended for training)
- 50GB+ disk space for model and dataset

### Setup

```bash
# Navigate to the granite directory
cd granite/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention 2 for faster training
pip install flash-attn --no-build-isolation
```

## Training

### Quick Test Run

Test the training pipeline with a small dataset:

```bash
python scripts/train.py --test_mode --test_size 100
```

### Full Training

```bash
python scripts/train.py
```

### Training with Custom Config

```bash
python scripts/train.py \
  --training_config configs/training_config.yaml \
  --model_config configs/model_config.yaml
```

### Training Options

```bash
python scripts/train.py --help
```

Key arguments:
- `--test_mode`: Use small subset for testing pipeline
- `--test_size`: Number of samples for test mode (default: 1000)
- `--no_wandb`: Disable Weights & Biases logging
- `--resume_from_checkpoint`: Resume from checkpoint directory

### Configuration

#### Training Configuration (`configs/training_config.yaml`)

Key parameters:
- `num_train_epochs`: 2 (default, adjust based on convergence)
- `per_device_train_batch_size`: 8 (adjust based on GPU memory)
- `gradient_accumulation_steps`: 8 (effective batch size = 64)
- `learning_rate`: 5e-5
- `max_seq_length`: 1024 (keeps 99.4% of dataset)

#### Model Configuration (`configs/model_config.yaml`)

LoRA settings (when `use_lora: true`):
- `lora_r`: 64 (LoRA rank)
- `lora_alpha`: 16 (scaling factor)  
- `lora_dropout`: 0.05
- Target modules: All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

Quantization (when `load_in_4bit: true`):
- 4-bit NF4 quantization with double quantization
- BFloat16 compute dtype

## Inference

### Interactive Mode

```bash
python scripts/inference.py \
  --model_path /data/outputs/granite-dutch-simplification \
  --interactive
```

### Single Text

```bash
python scripts/inference.py \
  --model_path /data/outputs/granite-dutch-simplification \
  --text "De minister-president kondigde gisteren tijdens een persconferentie een nieuw beleid aan."
```

### Batch Processing

```bash
python scripts/inference.py \
  --model_path /data/outputs/granite-dutch-simplification \
  --input_file input_texts.txt \
  --output_file simplified_texts.txt
```

### Inference Options

- `--max_new_tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--do_sample`: Use sampling instead of greedy decoding
- `--top_p`: Nucleus sampling parameter (default: 0.9)
- `--top_k`: Top-k sampling parameter
- `--repetition_penalty`: Penalty for repetition (default: 1.0)

## Model Architecture & Training Details

### Granite 3.3 2B Instruct

- **Base Model**: `ibm-granite/granite-3.3-2b-instruct`
- **Parameters**: ~2 billion parameters
- **Context Length**: 128K tokens (much longer than needed for most simplification tasks)
- **Architecture**: Transformer-based decoder-only model
- **Tokenizer**: Custom Granite tokenizer with chat template support

### Training Approach

The implementation supports two training modes:

1. **Full Fine-tuning** (`use_lora: false`, `load_in_4bit: false`)
   - Updates all model parameters
   - Requires more GPU memory (~16GB+ VRAM)
   - Generally produces better results

2. **LoRA/QLoRA Fine-tuning** (`use_lora: true`)
   - Parameter-efficient training with Low-Rank Adaptation
   - Optional 4-bit quantization for memory efficiency
   - Requires less GPU memory (~8GB VRAM with 4-bit)
   - Good results with much lower resource requirements

### Chat Template Format

Granite uses a specific chat template with role markers:

```
<|start_of_role|>system<|end_of_role|>[system prompt]<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Vereenvoudig: [original text]<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>[simplified text]<|end_of_text|>
```

### System Prompt

```
Je bent een AI-assistent die Nederlandse teksten vereenvoudigt naar een helder,
toegankelijk niveau voor iedereen, vergelijkbaar met de heldere taal die het
Jeugdjournaal gebruikt. Behoud de betekenis en belangrijke informatie, maar gebruik
eenvoudigere woorden en kortere zinnen. Schrijf niet kinderlijk, maar wel toegankelijk.
```

### Special Tokens

- **EOS Token**: `<|end_of_text|>` (also used as BOS and PAD)
- **Role Markers**: `<|start_of_role|>`, `<|end_of_role|>`

## Dataset

**Name**: `UWV/Leesplank_NL_wikipedia_simplifications_preprocessed`

The Leesplank dataset contains Wikipedia articles and their simplified versions in Dutch:

**Splits**:
- Train: ~1.89M samples
- Validation: ~540K samples  
- Test: ~269K samples

**Data Format**:
- `prompt`: Original complex text
- `result`: Simplified text
- `instruction`: "Vereenvoudig: " prefix

### Token Length Analysis

After applying Granite's chat template formatting (based on sample analysis):

**Input (System + User prompt):**
- Mean: ~240-245 tokens
- 95th percentile: ~405-410 tokens
- Max observed: ~1,880 tokens

**Output (Simplified text):**
- Mean: ~147-150 tokens
- 95th percentile: ~315 tokens
- 99th percentile: ~415-420 tokens
- Max observed: ~770 tokens

**Total (Full conversation):**
- Mean: ~390-395 tokens
- 95th percentile: ~710-720 tokens
- 99th percentile: ~940-960 tokens
- Max observed: ~2,320 tokens

### Data Filtering

Based on sequence length limits:
- `max_seq_length=512`: Keeps ~82% of data
- `max_seq_length=1024`: Keeps ~99.4% of data (recommended)
- `max_seq_length=2048`: Keeps 100% of data

## Training Pipeline

1. **Data Loading**: Load Leesplank dataset from Hugging Face Hub
2. **Chat Template Formatting**: Apply Granite's conversation format with system prompt
3. **Length Filtering**: Remove examples exceeding `max_seq_length` 
4. **Model Loading**: Load with optional quantization and LoRA configuration
5. **Training**: Supervised fine-tuning with TRL's SFTTrainer
6. **Monitoring**: Optional Weights & Biases integration for experiment tracking
7. **Checkpointing**: Regular model saves with best model selection

## License & Credits

This implementation is part of the UWV Leesplank project for Dutch text simplification.

### Components
- **Granite Model**: Apache 2.0 License (IBM Research)
- **Leesplank Dataset**: Available via Hugging Face Hub (UWV)
- **Implementation**: Uses Hugging Face Transformers and TRL

### Acknowledgments
- IBM Research for the Granite 3.3 model family
- UWV for creating and sharing the Leesplank dataset  
- Hugging Face for the transformers ecosystem
- The Dutch NLP research community
