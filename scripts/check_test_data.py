#!/usr/bin/env python3
"""Check if any test examples have zero-weight issues."""

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

print(f"Total examples: {len(examples)}")

# Simulate train/test split (same as in training)
# Shuffle with seed 42, take first 50 as test
import random
random.seed(42)
random.shuffle(examples)
test_examples = examples[:50]

print(f"Test examples: {len(test_examples)}")
print()

# Check each test example
zero_weight_count = 0
for i, row in enumerate(test_examples):
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
    datum_weights_sum = sum(datum.loss_fn_inputs["weights"].data)
    
    if datum_weights_sum == 0:
        zero_weight_count += 1
        print(f"❌ Test example {i}: ZERO WEIGHTS")
        print(f"   Prose: '{prose[:50]}...'")
        print(f"   Total tokens: {len(tokens)}, Prose tokens: {prose_len}")
        print()

print("="*60)
if zero_weight_count > 0:
    print(f"Found {zero_weight_count} test examples with zero weights!")
    print("This will cause NaN in test loss!")
else:
    print("All test examples have valid weights.")
    print("NaN must be coming from elsewhere...")

