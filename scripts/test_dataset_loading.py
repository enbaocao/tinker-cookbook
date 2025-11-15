#!/usr/bin/env python3
"""Test that the dataset loading works correctly with the fix."""

import sys
import asyncio
from train_prose_simple import build_config_blueprint

# Get the config
blueprint = build_config_blueprint()
config = blueprint.make()

# Build the datasets
train_dataset, test_dataset = config.dataset_builder()

print(f"Train dataset:")
print(f"  Length (num batches): {len(train_dataset)}")
print(f"  Batch size: {train_dataset.batch_size}")
print(f"  Total examples: ~{len(train_dataset) * train_dataset.batch_size}")
print()

if test_dataset is not None:
    print(f"Test dataset:")
    print(f"  Length (num batches): {len(test_dataset)}")
    print(f"  Batch size: {test_dataset.batch_size}")
    print(f"  Total examples: ~{len(test_dataset) * test_dataset.batch_size}")
    print()

    # Try to get the first (and only) batch
    try:
        test_batch = test_dataset.get_batch(0)
        print(f"✅ Successfully loaded test batch!")
        print(f"  Number of datums in batch: {len(test_batch)}")

        # Check weights
        total_weights = sum(sum(datum.loss_fn_inputs["weights"].data) for datum in test_batch)
        print(f"  Total weights across batch: {total_weights}")

        if total_weights > 0:
            print(f"✅ Test dataset has non-zero weights!")
            print(f"   NaN issue should be FIXED!")
        else:
            print(f"❌ Test dataset still has zero weights!")
    except Exception as e:
        print(f"❌ Error loading test batch: {e}")
else:
    print("❌ No test dataset created!")
