"""
Inference script for Dutch text simplification using finetuned Granite model.

This script provides both interactive and batch inference capabilities.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import List, Dict

import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_utils import load_model_for_inference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with finetuned Granite model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the finetuned model"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to simplify"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to file with texts to simplify (one per line)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save simplified texts"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt (uses default if not specified)"
    )

    return parser.parse_args()


DEFAULT_SYSTEM_PROMPT = (
    "Je bent een AI-assistent die Nederlandse teksten vereenvoudigt naar een "
    "helder, toegankelijk niveau voor iedereen, vergelijkbaar met de heldere taal "
    "die het Jeugdjournaal gebruikt. Behoud de betekenis en belangrijke informatie, "
    "maar gebruik eenvoudigere woorden en kortere zinnen. Schrijf niet kinderlijk, "
    "maar wel toegankelijk."
)


def simplify_text(
    text: str,
    model,
    tokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9
) -> str:
    """
    Simplify a single text using the model.

    Args:
        text: Text to simplify
        model: The finetuned model
        tokenizer: The tokenizer
        system_prompt: System instruction
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        top_p: Nucleus sampling parameter

    Returns:
        Simplified text
    """
    # Create conversation
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Vereenvoudig: {text}"}
    ]

    # Apply chat template with generation prompt
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated part (exclude the prompt)
    generated_tokens = outputs[0][len(inputs.input_ids[0]):]
    simplified = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return simplified.strip()


def simplify_batch(
    texts: List[str],
    model,
    tokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
    batch_size: int = 4
) -> List[str]:
    """
    Simplify multiple texts in batches.

    Args:
        texts: List of texts to simplify
        model: The finetuned model
        tokenizer: The tokenizer
        system_prompt: System instruction
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        top_p: Nucleus sampling parameter
        batch_size: Batch size for processing

    Returns:
        List of simplified texts
    """
    simplified_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")

        batch_simplified = []
        for text in batch:
            simplified = simplify_text(
                text=text,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p
            )
            batch_simplified.append(simplified)

        simplified_texts.extend(batch_simplified)

    return simplified_texts


def interactive_mode(model, tokenizer, args):
    """Run interactive simplification."""
    print("\n" + "="*60)
    print("Interactive Dutch Text Simplification")
    print("="*60)
    print("Enter text to simplify (or 'quit' to exit)")
    print("="*60 + "\n")

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    while True:
        try:
            text = input("\nOriginal text: ").strip()

            if text.lower() in ["quit", "exit", "q"]:
                print("\nExiting...")
                break

            if not text:
                continue

            print("\nSimplifying...")
            simplified = simplify_text(
                text=text,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                top_p=args.top_p
            )

            print(f"\nSimplified text: {simplified}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    # Model path can be either local or HuggingFace model identifier
    # if not os.path.exists(args.model_path):
    #     raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    try:
        model, tokenizer = load_model_for_inference(
            model_path=args.model_path,
            use_flash_attention=True,
            load_in_4bit=True
        )
    except Exception as e:
        print(f"Warning: Could not load with 4-bit quantization: {e}")
        print("Attempting to load without quantization...")
        model, tokenizer = load_model_for_inference(
            model_path=args.model_path,
            use_flash_attention=False,
            load_in_4bit=False
        )
    print("Model loaded successfully!")

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, args)
        return

    # Single text
    if args.text:
        print(f"\nOriginal: {args.text}")
        simplified = simplify_text(
            text=args.text,
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p
        )
        print(f"Simplified: {simplified}\n")
        return

    # Batch from file
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        print(f"Reading texts from {args.input_file}...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Simplifying {len(texts)} texts...")
        simplified_texts = simplify_batch(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p
        )

        # Save to file if specified
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for simplified in simplified_texts:
                    f.write(simplified + '\n')
            print(f"\nSimplified texts saved to {args.output_file}")
        else:
            # Print results
            print("\nResults:")
            print("="*60)
            for i, (original, simplified) in enumerate(zip(texts, simplified_texts), 1):
                print(f"\n{i}. Original: {original}")
                print(f"   Simplified: {simplified}")
                print("-"*60)

        return

    # No mode specified
    print("Error: Please specify one of: --text, --input_file, or --interactive")
    print("Use --help for more information")


if __name__ == "__main__":
    main()
