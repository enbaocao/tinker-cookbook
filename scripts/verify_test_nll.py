#!/usr/bin/env python3
"""Verify that test NLL computation won't produce NaN."""

import json
import torch
import random
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.common import datum_from_tokens_weights, compute_mean_nll
import tinker

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = get_tokenizer(model_name)

# Load and split data exactly as in training
examples = []
with open("example-data/claude_labeled.jsonl") as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

print(f"Total examples: {len(examples)}")

# Shuffle with same seed as training
random.seed(42)
random.shuffle(examples)
test_examples = examples[:50]

print(f"Test examples: {len(test_examples)}")
print()

# Create datums for all test examples
test_datums = []
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
    test_datums.append(datum)

# Extract weights from all datums
all_weights = [datum.loss_fn_inputs["weights"] for datum in test_datums]

# Check total weights
total_weights = sum(sum(w.data) for w in all_weights)
print(f"Total weights across all test examples: {total_weights}")
print()

if total_weights == 0:
    print("❌ ERROR: Total weights is ZERO!")
    print("This WILL cause NaN in test loss!")
else:
    print("✅ SUCCESS: Total weights is non-zero")
    print("Test loss computation should work correctly")
    print()

    # Simulate what compute_mean_nll would do
    # (We can't actually call it without real logprobs, but we can check the weights)
    print("Simulating NLL computation:")
    print(f"  - Number of test datums: {len(test_datums)}")
    print(f"  - Total weighted tokens: {total_weights:.0f}")
    print(f"  - Average tokens per example: {total_weights/len(test_datums):.2f}")
    print()

    # Create dummy logprobs to verify compute_mean_nll works
    dummy_logprobs = []
    for datum in test_datums:
        weights_data = datum.loss_fn_inputs["weights"]
        # Create dummy logprobs of same shape
        dummy_logprob_data = [-1.0] * len(weights_data.data)  # Dummy values
        dummy_logprobs.append(
            tinker.TensorData(
                data=dummy_logprob_data,
                dtype="float32",
                shape=weights_data.shape,
            )
        )

    # Test compute_mean_nll with dummy data
    try:
        test_nll = compute_mean_nll(dummy_logprobs, all_weights)
        print(f"✅ compute_mean_nll succeeded with dummy data")
        print(f"   Result: {test_nll:.4f} (expected ~1.0 with dummy logprobs of -1.0)")

        if test_nll != test_nll:  # Check for NaN
            print("❌ ERROR: Result is NaN!")
        else:
            print("✅ Result is not NaN!")
    except Exception as e:
        print(f"❌ ERROR in compute_mean_nll: {e}")

print()
print("="*60)
print("CONCLUSION:")
if total_weights > 0:
    print("Test set is ready to be re-enabled!")
    print("NaN issue should be resolved.")
else:
    print("DO NOT re-enable test set - NaN issue persists!")
