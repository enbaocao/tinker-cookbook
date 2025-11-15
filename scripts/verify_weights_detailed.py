#!/usr/bin/env python3
"""Detailed verification of weight assignment for test examples."""

import json
import torch
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.common import datum_from_tokens_weights

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = get_tokenizer(model_name)

# Load all examples
examples = []
with open("example-data/claude_labeled.jsonl") as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

# Simulate train/test split (same as in training)
import random
random.seed(42)
random.shuffle(examples)
test_examples = examples[:50]

print(f"Checking {len(test_examples)} test examples...")
print()

# Check a few examples in detail
for i in range(min(3, len(test_examples))):
    row = test_examples[i]
    messages = row["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assistant_msg = next(m for m in messages if m["role"] == "assistant")

    prompt = user_msg["content"]
    prose = assistant_msg["content"]

    text = f"{prompt}\n\n{prose}"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    prose_tokens = tokenizer.encode(prose, add_special_tokens=False)
    prose_len = len(prose_tokens)

    tokens_tensor = torch.tensor(tokens)
    weights = torch.zeros(len(tokens))
    if prose_len > 0:
        weights[-prose_len:] = 1.0

    # Create datum
    datum = datum_from_tokens_weights(tokens_tensor, weights, max_length=2048)
    datum_weights = datum.loss_fn_inputs["weights"].data
    datum_weights_sum = sum(datum_weights)

    print(f"Example {i}:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Prose tokens: {prose_len}")
    print(f"  Weight sum (before datum_from_tokens_weights): {weights.sum()}")
    print(f"  Weight sum (after datum_from_tokens_weights): {datum_weights_sum}")
    print(f"  Prose preview: '{prose[:60]}...'")
    print()

# Check all examples for statistics
all_weight_sums = []
for row in test_examples:
    messages = row["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assistant_msg = next(m for m in messages if m["role"] == "assistant")

    prompt = user_msg["content"]
    prose = assistant_msg["content"]

    text = f"{prompt}\n\n{prose}"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    prose_tokens = tokenizer.encode(prose, add_special_tokens=False)
    prose_len = len(prose_tokens)

    tokens_tensor = torch.tensor(tokens)
    weights = torch.zeros(len(tokens))
    if prose_len > 0:
        weights[-prose_len:] = 1.0

    datum = datum_from_tokens_weights(tokens_tensor, weights, max_length=2048)
    datum_weights_sum = sum(datum.loss_fn_inputs["weights"].data)
    all_weight_sums.append(datum_weights_sum)

print("="*60)
print("Statistics across all test examples:")
print(f"  Min weight sum: {min(all_weight_sums)}")
print(f"  Max weight sum: {max(all_weight_sums)}")
print(f"  Mean weight sum: {sum(all_weight_sums) / len(all_weight_sums):.2f}")
print(f"  Examples with zero weights: {sum(1 for w in all_weight_sums if w == 0)}")
