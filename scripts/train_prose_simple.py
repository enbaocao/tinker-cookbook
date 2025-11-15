#!/usr/bin/env python3
"""
Train a prose-writing LoRA using a simplified approach.
Treats prompt→prose as pure text completion without chat formatting.

Usage:
    python train_prose_simple.py
"""

import chz
import sys
import asyncio
import json
import torch
import blobfile
import datasets
from tinker_cookbook import cli_utils, hyperparam_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer
import tinker


@chz.chz
class SimpleProseDatasetBuilder(SupervisedDatasetBuilder):
    """
    Simple dataset builder that treats prompt+prose as plain text completion.
    No chat formatting - just: prompt\\n\\nprose
    """
    file_path: str
    model_name_for_tokenizer: str
    batch_size: int
    test_size: int = 50
    shuffle_seed: int = 42
    max_length: int = 2048
    
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSONL
        conversations = []
        with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                conversations.append(data)
        
        # Create dataset
        dataset = datasets.Dataset.from_list(conversations)
        
        # Shuffle
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)
        
        # Split
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            train_ds = dataset
            test_ds = None
        
        # Get tokenizer
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        
        def map_fn(row: dict) -> tinker.Datum:
            """Convert a conversation to a simple completion task."""
            messages = row["messages"]
            
            # Extract prompt and prose
            user_msg = next(m for m in messages if m["role"] == "user")
            assistant_msg = next(m for m in messages if m["role"] == "assistant")
            
            prompt = user_msg["content"]
            prose = assistant_msg["content"]
            
            # Format as simple text: prompt\n\nprose
            text = f"{prompt}\n\n{prose}"
            
            # Tokenize full text with BOS
            tokens = tokenizer.encode(text, add_special_tokens=True)
            tokens_tensor = torch.tensor(tokens)
            
            # Tokenize just the prose to find where it starts
            # (Don't add BOS to this one)
            prose_tokens = tokenizer.encode(prose, add_special_tokens=False)
            prose_len = len(prose_tokens)
            
            # Create weights: 0 for prompt + separator, 1 for prose
            # The prose tokens should be at the END of the sequence
            weights = torch.zeros(len(tokens))
            if prose_len > 0:
                weights[-prose_len:] = 1.0
            
            # Use the standard helper to create datum with proper input/target split
            return datum_from_tokens_weights(tokens_tensor, weights, self.max_length)
        
        # Wrap in supervised dataset
        from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
        
        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.batch_size, map_fn=map_fn
        )

        test_dataset = None
        if test_ds is not None:
            # Use len(test_ds) as batch_size to ensure all test examples are in one batch
            # (avoids dropping test data when test_size < batch_size)
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )
        
        return train_dataset, test_dataset


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Build training configuration for prose LoRA (simplified version)."""
    model_name = "meta-llama/Llama-3.1-8B"
    learning_rate = hyperparam_utils.get_lr(model_name, is_lora=True)
    
    dataset = SimpleProseDatasetBuilder(
        file_path="example-data/claude_labeled.jsonl",
        model_name_for_tokenizer=model_name,
        batch_size=64,
        test_size=50,  # Use 50 examples for test set
        shuffle_seed=42,
        max_length=2048,
    )
    
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/tinker-prose-full",
            "model_name": model_name,
            "dataset_builder": dataset,
            "learning_rate": learning_rate,
            "lora_rank": 32,
            "lr_schedule": "linear",
            "num_epochs": 3,  # 3 epochs to avoid overfitting
            "eval_every": 16,
            "save_every": 100,
            # Weights & Biases logging
            "wandb_project": "prose-lora",
            "wandb_name": None,  # Auto-generated based on timestamp
        }
    )


def main(config: train.Config):
    """Main training entry point."""
    print("="*60)
    print("Simple Prose LoRA Training (No Chat Formatting)")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Learning rate: {config.learning_rate:.2e}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Log path: {config.log_path}")
    print("="*60)
    print()
    
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())

