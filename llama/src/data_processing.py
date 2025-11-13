"""
Data processing utilities for Dutch text simplification training.

This module handles loading and formatting the Leesplank dataset
into the Granite chat template format.
"""

from datasets import load_dataset, DatasetDict
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_chat_template(
    original_text: str,
    simplified_text: str,
    system_prompt: str,
    tokenizer
) -> str:
    """
    Format a single example into Granite's chat template format.

    Args:
        original_text: The original text to be simplified
        simplified_text: The simplified version of the text
        system_prompt: System instruction for the model
        tokenizer: The Granite tokenizer with chat template

    Returns:
        Formatted conversation string ready for training
    """
    # Create conversation in the format expected by Granite
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Vereenvoudig: {original_text}"},
        {"role": "assistant", "content": simplified_text}
    ]

    # Apply the chat template
    # tokenize=False returns the formatted string without tokenizing
    formatted_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )

    return formatted_text


def load_and_prepare_dataset(
    dataset_name: str,
    tokenizer,
    system_prompt: str,
    max_seq_length: int = 2048,
    num_proc: int = 4,
    test_size: Optional[int] = None
) -> DatasetDict:
    """
    Load the Leesplank dataset and format it for training.

    Args:
        dataset_name: HuggingFace dataset identifier
        tokenizer: The model tokenizer
        system_prompt: System instruction for text simplification
        max_seq_length: Maximum sequence length for filtering
        num_proc: Number of processes for parallel processing
        test_size: If specified, use only this many samples (for testing)

    Returns:
        Formatted dataset with train/val/test splits
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # If test_size is specified, take a smaller subset
    if test_size:
        logger.info(f"Using test mode with {test_size} samples per split")
        dataset = DatasetDict({
            "train": dataset["train"].select(range(min(test_size, len(dataset["train"])))),
            "val": dataset["val"].select(range(min(test_size, len(dataset["val"])))),
            "test": dataset["test"].select(range(min(test_size, len(dataset["test"]))))
        })

    logger.info(f"Dataset sizes - Train: {len(dataset['train'])}, "
                f"Val: {len(dataset['val'])}, Test: {len(dataset['test'])}")

    def format_example(example: Dict) -> Dict:
        """Format a single example."""
        # The dataset has 'prompt' (original) and 'result' (simplified)
        original = example['prompt']
        simplified = example['result']

        # Format using chat template
        formatted = format_chat_template(
            original_text=original,
            simplified_text=simplified,
            system_prompt=system_prompt,
            tokenizer=tokenizer
        )

        return {
            "formatted_text": formatted
        }

    # Format all examples
    logger.info("Formatting dataset with chat template...")
    formatted_dataset = dataset.map(
        format_example,
        num_proc=num_proc,
        desc="Formatting examples",
        remove_columns=dataset["train"].column_names  # Remove original columns to avoid confusion
    )

    # Filter out examples that are too long
    def filter_length(example: Dict) -> bool:
        """Filter examples by tokenized length."""
        tokenized = tokenizer(example["formatted_text"], truncation=False)
        return len(tokenized["input_ids"]) <= max_seq_length

    logger.info("Filtering examples by length...")
    filtered_dataset = formatted_dataset.filter(
        filter_length,
        num_proc=num_proc,
        desc="Filtering by length"
    )

    logger.info(f"After filtering - Train: {len(filtered_dataset['train'])}, "
                f"Val: {len(filtered_dataset['val'])}, Test: {len(filtered_dataset['test'])}")

    # Calculate filtering statistics
    for split in ["train", "val", "test"]:
        original_size = len(formatted_dataset[split])
        filtered_size = len(filtered_dataset[split])
        filtered_out = original_size - filtered_size
        pct = (filtered_out / original_size * 100) if original_size > 0 else 0
        logger.info(f"{split}: Filtered out {filtered_out} examples ({pct:.2f}%)")

    # Randomly sample validation and test sets to desired sizes
    # Using seed for reproducibility
    VAL_SAMPLE_SIZE = 1_000
    TEST_SAMPLE_SIZE = 10_000
    SEED = 42

    # Use all training data
    logger.info(f"Using all {len(filtered_dataset['train'])} training records")

    # Sample validation set
    if len(filtered_dataset["val"]) > VAL_SAMPLE_SIZE:
        logger.info(f"Sampling validation set from {len(filtered_dataset['val'])} to {VAL_SAMPLE_SIZE} records")
        filtered_dataset["val"] = filtered_dataset["val"].shuffle(seed=SEED).select(range(VAL_SAMPLE_SIZE))
    else:
        logger.info(f"Validation set has {len(filtered_dataset['val'])} records (no sampling needed)")

    # Sample test set
    if len(filtered_dataset["test"]) > TEST_SAMPLE_SIZE:
        logger.info(f"Sampling test set from {len(filtered_dataset['test'])} to {TEST_SAMPLE_SIZE} records")
        filtered_dataset["test"] = filtered_dataset["test"].shuffle(seed=SEED).select(range(TEST_SAMPLE_SIZE))
    else:
        logger.info(f"Test set has {len(filtered_dataset['test'])} records (no sampling needed)")

    logger.info(f"Final dataset sizes - Train: {len(filtered_dataset['train'])}, "
                f"Val: {len(filtered_dataset['val'])}, Test: {len(filtered_dataset['test'])}")

    return filtered_dataset


def get_formatting_func(tokenizer, system_prompt: str):
    """
    Get a formatting function for use with SFTTrainer.

    This is an alternative to preprocessing the entire dataset.
    Useful for streaming or on-the-fly formatting.

    Args:
        tokenizer: The model tokenizer
        system_prompt: System instruction for text simplification

    Returns:
        Function that formats examples
    """
    def formatting_func(example: Dict) -> Dict[str, List[str]]:
        """Format examples on the fly."""
        # Handle both single examples and batches
        if isinstance(example['prompt'], list):
            # Batch processing
            formatted_texts = []
            for orig, simp in zip(example['prompt'], example['result']):
                formatted = format_chat_template(
                    original_text=orig,
                    simplified_text=simp,
                    system_prompt=system_prompt,
                    tokenizer=tokenizer
                )
                formatted_texts.append(formatted)
            return {
                "formatted_text": formatted_texts
            }
        else:
            # Single example
            formatted = format_chat_template(
                original_text=example['prompt'],
                simplified_text=example['result'],
                system_prompt=system_prompt,
                tokenizer=tokenizer
            )
            return {
                "formatted_text": [formatted]
            }

    return formatting_func


if __name__ == "__main__":
    # Test the data processing
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b-Instruct")

    # Test formatting
    system_prompt = ("Je bent een AI-assistent die Nederlandse teksten vereenvoudigt naar een "
                    "helder, toegankelijk niveau voor iedereen, vergelijkbaar met de heldere taal "
                    "die het Jeugdjournaal gebruikt. Behoud de betekenis en belangrijke informatie, "
                    "maar gebruik eenvoudigere woorden en kortere zinnen. Schrijf niet kinderlijk, "
                    "maar wel toegankelijk.")

    original = "De minister-president kondigde gisteren een nieuw beleid aan."
    simplified = "De premier maakte gisteren een nieuw plan bekend."

    formatted = format_chat_template(original, simplified, system_prompt, tokenizer)
    print("Formatted example:")
    print(formatted)
    print("\n" + "="*80 + "\n")

    # Test dataset loading (with small sample)
    dataset = load_and_prepare_dataset(
        "UWV/Leesplank_NL_wikipedia_simplifications_preprocessed",
        tokenizer,
        system_prompt,
        test_size=10
    )

    print("Sample from dataset:")
    print(dataset["train"][0]["formatted_text"])