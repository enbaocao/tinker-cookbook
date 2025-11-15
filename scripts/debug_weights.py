#!/usr/bin/env python3
"""Debug script to check if weights are being set correctly."""

import json
import torch
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.common import datum_from_tokens_weights

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = get_tokenizer(model_name)

# Load one example
with open("example-data/claude_labeled.jsonl") as f:
    row = json.loads(f.readline())

messages = row["messages"]
user_msg = next(m for m in messages if m["role"] == "user")
assistant_msg = next(m for m in messages if m["role"] == "assistant")

prompt = user_msg["content"]
prose = assistant_msg["content"]

print("="*60)
print("PROMPT:")
print(prompt)
print("\nPROSE:")
print(prose)
print("="*60)

# Test the current approach
text = f"{prompt}\n\n{prose}"
tokens = tokenizer.encode(text, add_special_tokens=True)
tokens_tensor = torch.tensor(tokens)

prose_tokens = tokenizer.encode(prose, add_special_tokens=False)
prose_len = len(prose_tokens)

weights = torch.zeros(len(tokens))
if prose_len > 0:
    weights[-prose_len:] = 1.0

print(f"\nTotal tokens: {len(tokens)}")
print(f"Prose token count: {prose_len}")
print(f"Weights sum: {weights.sum().item()}")
print(f"Weights > 0: {(weights > 0).sum().item()}")

# Create datum and check
datum = datum_from_tokens_weights(tokens_tensor, weights, None)
datum_weights = datum.loss_fn_inputs["weights"]
print(f"\nDatum weights sum: {sum(datum_weights.data)}")
print(f"Datum weights shape: {datum_weights.shape}")

# Show last few tokens
print("\nLast 10 tokens (should be prose):")
for i in range(max(0, len(tokens)-10), len(tokens)):
    token_id = tokens[i]
    token_text = tokenizer.decode([token_id])
    weight = weights[i].item()
    print(f"  {i}: '{token_text}' (id={token_id}, weight={weight})")

