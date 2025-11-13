"""
Model utilities for loading and configuring Granite models.

This module handles model and tokenizer loading with proper
quantization and LoRA configuration.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes configuration for quantization.

    Args:
        load_in_4bit: Whether to use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4')
        bnb_4bit_use_double_quant: Whether to use double quantization

    Returns:
        BitsAndBytesConfig object
    """
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }

    compute_dtype = compute_dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
    )

    return bnb_config


def create_lora_config(
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    lora_target_modules: Optional[list] = None
) -> LoraConfig:
    """
    Create LoRA configuration for parameter-efficient finetuning.

    Args:
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability for LoRA layers
        lora_bias: Bias training strategy ('none', 'all', or 'lora_only')
        lora_target_modules: List of module names to apply LoRA to

    Returns:
        LoraConfig object
    """
    if lora_target_modules is None:
        # Default target modules for Granite (similar to LLaMA architecture)
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM"
    )

    return lora_config


def load_model_and_tokenizer(
    model_name: str,
    model_config: Dict,
    use_flash_attention: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with proper configuration.

    Args:
        model_name: HuggingFace model identifier
        model_config: Dictionary with model configuration
        use_flash_attention: Whether to use Flash Attention 2

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer from {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
        use_fast=model_config.get("use_fast_tokenizer", True),
        padding_side=model_config.get("padding_side", "right")
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Create BitsAndBytes config for quantization
    bnb_config = None
    if model_config.get("load_in_4bit", False):
        logger.info("Creating 4-bit quantization config")
        bnb_config = create_bnb_config(
            load_in_4bit=model_config["load_in_4bit"],
            bnb_4bit_compute_dtype=model_config.get("bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True)
        )

    # Determine attention implementation
    attn_implementation = None
    if use_flash_attention and model_config.get("attn_implementation") == "flash_attention_2":
        attn_implementation = "flash_attention_2"
        logger.info("Using Flash Attention 2")

    # Map torch dtype
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto"
    }
    torch_dtype = torch_dtype_map.get(
        model_config.get("torch_dtype", "bfloat16"),
        torch.bfloat16
    )

    logger.info(f"Loading model from {model_name}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", False),
        attn_implementation=attn_implementation,
        use_cache=model_config.get("use_cache", False)
    )

    # Prepare model for k-bit training if using quantization
    if bnb_config is not None:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA if configured
    if model_config.get("use_lora", True):
        logger.info("Applying LoRA configuration")
        lora_config = create_lora_config(
            lora_r=model_config.get("lora_r", 64),
            lora_alpha=model_config.get("lora_alpha", 16),
            lora_dropout=model_config.get("lora_dropout", 0.05),
            lora_bias=model_config.get("lora_bias", "none"),
            lora_target_modules=model_config.get("lora_target_modules")
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_model_for_inference(
    model_path: str,
    use_flash_attention: bool = True,
    load_in_4bit: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a finetuned model for inference.

    Args:
        model_path: Path to the finetuned model
        use_flash_attention: Whether to use Flash Attention 2
        load_in_4bit: Whether to load in 4-bit

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model for inference from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create quantization config if needed
    bnb_config = None
    if load_in_4bit:
        bnb_config = create_bnb_config()

    # Determine attention implementation
    attn_implementation = "flash_attention_2" if use_flash_attention else None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation
    )

    model.eval()

    return model, tokenizer


if __name__ == "__main__":
    # Test model loading
    import yaml

    with open("configs/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(
        "meta-llama/Llama-3.2-3b-Instruct",
        model_config
    )

    print(f"Model loaded successfully")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")