#!/usr/bin/env python3
"""
Train a prose-writing LoRA using labeled data from the prose pipeline.

Usage:
    python train_prose_lora.py file_path=training_data.jsonl

Optional overrides:
    python train_prose_lora.py \
        file_path=training_data.jsonl \
        model_name=meta-llama/Llama-3.1-8B \
        num_epochs=3 \
        lora_rank=64 \
        batch_size=32
"""

import chz
import sys
import asyncio
from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """
    Build training configuration for prose LoRA.

    Key settings:
    - TrainOnWhat.ALL_ASSISTANT_MESSAGES: Trains only on prose, not prompts
    - Learning rate auto-calculated for model + LoRA
    - LoRA rank 32 (increase to 64 for more complex styles)
    """
    # Default model - good balance of quality and training speed
    model_name = "meta-llama/Llama-3.1-8B"

    # Auto-select best renderer for this model
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    # Calculate optimal learning rate for this model with LoRA
    # LoRA typically needs ~10x higher LR than full fine-tuning
    learning_rate = hyperparam_utils.get_lr(model_name, is_lora=True)

    # Dataset configuration
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=2048,  # Adjust if your prose is longer
        batch_size=64,    # Adjust based on dataset size
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,  # Train on prose only!
    )

    # Load data from JSONL file
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path="example-data/claude_labeled.jsonl",  # Override with dataset_builder.file_path=...
        test_size=50,      # Hold out 50 examples for evaluation
        shuffle_seed=42,   # Reproducible shuffling
    )

    # Training configuration
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/tinker-prose-lora",
            "model_name": model_name,
            "dataset_builder": dataset,
            "learning_rate": learning_rate,  # Auto-calculated optimal LR
            "lora_rank": 32,          # Higher = more capacity (try 64 for complex styles)
            "lr_schedule": "linear",  # Linear decay over training
            "num_epochs": 5,          # 3-5 epochs typical for prose
            "eval_every": 16,         # Evaluate every N batches
            "save_every": 100,        # Save checkpoints frequently
        }
    )


def main(config: train.Config):
    """
    Main training entry point.

    Checks log directory for existing runs, then starts training.
    """
    print("="*60)
    print("Prose LoRA Training")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Learning rate: {config.learning_rate:.2e}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Log path: {config.log_path}")
    print("="*60)
    print()

    # Avoid clobbering previous runs - asks user what to do
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    # Start async training
    asyncio.run(train.main(config))


if __name__ == "__main__":
    # Build config blueprint
    blueprint = build_config_blueprint()

    # Apply command-line overrides
    # Example: python train_prose_lora.py file_path=my_data.jsonl num_epochs=5
    blueprint.make_from_argv(sys.argv[1:])

    # Run training
    main(blueprint.make())
