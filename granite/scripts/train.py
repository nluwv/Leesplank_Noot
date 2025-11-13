"""
Training script for finetuning Granite 3.3 2B on Dutch text simplification.

This script uses TRL's SFTTrainer for supervised fine-tuning with QLoRA.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_utils import load_model_and_tokenizer
from src.data_processing import load_and_prepare_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune Granite 3.3 2B for Dutch text simplification"
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode with small dataset"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of samples to use in test mode"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Running as part of wandb sweep (parameters come from wandb.config)"
    )

    return parser.parse_args()


def load_configs(training_config_path: str, model_config_path: str):
    """Load configuration files."""
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    return training_config, model_config


def setup_training_args(training_config: dict, no_wandb: bool = False, sweep_config: dict = None):
    """Create TrainingArguments from config.

    Args:
        training_config: Base training configuration
        no_wandb: Whether to disable wandb logging
        sweep_config: Optional sweep parameters that override training_config
    """
    # Update report_to based on wandb setting
    report_to = training_config.get("report_to", ["tensorboard"])
    if no_wandb and "wandb" in report_to:
        report_to.remove("wandb")

    # Override with sweep config if provided
    if sweep_config:
        training_config = {**training_config, **sweep_config}

    # Determine if using max_steps or num_train_epochs
    max_steps = training_config.get("max_steps", -1)
    num_train_epochs = training_config.get("num_train_epochs", 3) if max_steps == -1 else None

    # Determine warmup strategy (warmup_steps takes precedence over warmup_ratio)
    warmup_steps = training_config.get("warmup_steps", 0)
    warmup_ratio = training_config.get("warmup_ratio", 0.03) if warmup_steps == 0 else 0.0

    # Create training arguments
    training_args = SFTConfig(
        output_dir=training_config["output_dir"],
        max_steps=max_steps,
        num_train_epochs=num_train_epochs if num_train_epochs else 1,
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        optim=training_config.get("optim", "paged_adamw_8bit"),
        logging_steps=training_config.get("logging_steps", 10),
        eval_strategy=training_config.get("eval_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        bf16=training_config.get("bf16", True),
        tf32=training_config.get("tf32", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        report_to=report_to,
        seed=training_config.get("seed", 42),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        # SFT-specific settings
        max_length=training_config.get("max_seq_length", 2048),
        packing=training_config.get("packing", False),
        dataset_text_field=training_config.get("dataset_text_field", "formatted_text"),
    )

    return training_args


def train(args):
    """Main training function."""
    logger.info("="*60)
    logger.info("Granite Dutch Text Simplification Training")
    logger.info("="*60)

    # Load configurations
    logger.info("Loading configurations...")
    training_config, model_config = load_configs(
        args.training_config,
        args.model_config
    )

    # Override test mode settings
    test_size = None
    if args.test_mode:
        logger.info(f"Running in TEST MODE with {args.test_size} samples")
        test_size = args.test_size
        training_config["num_train_epochs"] = 1
        training_config["save_steps"] = 50
        training_config["eval_steps"] = 50

    # Initialize wandb if not disabled
    sweep_config = None
    if not args.no_wandb and "wandb" in training_config.get("report_to", []):
        if args.sweep:
            # In sweep mode, initialize wandb (it will pick up sweep config automatically)
            logger.info("Running in SWEEP MODE - using wandb.config parameters")
            wandb.init()
            sweep_config = dict(wandb.config)
            logger.info(f"Sweep parameters: {sweep_config}")

            # Convert sweep parameters to correct types (wandb passes them as strings)
            if 'learning_rate' in sweep_config:
                sweep_config['learning_rate'] = float(sweep_config['learning_rate'])
            if 'per_device_train_batch_size' in sweep_config:
                sweep_config['per_device_train_batch_size'] = int(sweep_config['per_device_train_batch_size'])
            if 'max_steps' in sweep_config:
                sweep_config['max_steps'] = int(sweep_config['max_steps'])
            if 'eval_steps' in sweep_config:
                sweep_config['eval_steps'] = int(sweep_config['eval_steps'])
            if 'warmup_steps' in sweep_config:
                sweep_config['warmup_steps'] = int(sweep_config['warmup_steps'])
        else:
            # Normal mode - initialize wandb
            wandb.init(
                project="granite-dutch-simplification",
                name=f"granite-3.3-2b-{training_config['num_train_epochs']}ep",
                config={**training_config, **model_config}
            )

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=training_config["model_name"],
        model_config=model_config,
        use_flash_attention=True
    )

    # Load and prepare dataset
    logger.info("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(
        dataset_name=training_config["dataset_name"],
        tokenizer=tokenizer,
        system_prompt=training_config["system_prompt"],
        max_seq_length=training_config["max_seq_length"],
        num_proc=4,
        test_size=test_size
    )

    # Setup training arguments
    logger.info("Setting up training arguments...")
    training_args = setup_training_args(training_config, args.no_wandb, sweep_config)

    # Create trainer
    logger.info("Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
    )

    # Log training info
    logger.info(f"Training samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['val'])}")
    logger.info(f"Test samples: {len(dataset['test'])}")
    logger.info(f"Output directory: {training_config['output_dir']}")
    if training_args.max_steps > 0:
        logger.info(f"Max steps: {training_args.max_steps}")
    else:
        logger.info(f"Number of epochs: {training_config['num_train_epochs']}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Warmup steps: {training_args.warmup_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    # Start training
    logger.info("Starting training...")
    logger.info("="*60)

    try:
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Saving checkpoint before exiting...")
        trainer.save_model(os.path.join(training_config["output_dir"], "interrupted"))
        return

    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)

    # Save the final model
    logger.info("Saving final model...")
    final_output_dir = os.path.join(training_config["output_dir"], "final_model")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    logger.info(f"Model saved to: {final_output_dir}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")

    # Tokenize test dataset for evaluation
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_text"],
            truncation=True,
            max_length=training_config["max_seq_length"],
            padding=False
        )

    test_dataset_tokenized = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing test set"
    )

    test_results = trainer.evaluate(test_dataset_tokenized)
    logger.info(f"Test results: {test_results}")

    # Save test results
    results_file = os.path.join(training_config["output_dir"], "test_results.yaml")
    with open(results_file, 'w') as f:
        yaml.dump(test_results, f)

    logger.info(f"Test results saved to: {results_file}")

    # Finish wandb run
    if not args.no_wandb and "wandb" in training_config.get("report_to", []):
        wandb.finish()

    logger.info("Training pipeline completed successfully!")


def main():
    """Main entry point."""
    args = parse_args()

    # Verify config files exist
    if not os.path.exists(args.training_config):
        raise FileNotFoundError(f"Training config not found: {args.training_config}")
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config not found: {args.model_config}")

    # Run training
    train(args)


if __name__ == "__main__":
    main()
