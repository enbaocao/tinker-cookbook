#!/usr/bin/env python3
"""
Sample from a trained prose LoRA model.

Usage:
    python scripts/sample_prose.py
"""

import asyncio
from tinker import SamplingClient

# Your checkpoint from training
model_path = "tinker://98d93e3c-0eef-5fd2-bb6d-fa38952b4bc0:train:0/sampler_weights/final"


async def sample():
    print("Initializing sampling client...")
    client = SamplingClient(
        model_path=model_path,
        renderer_name="llama3",  # matches your training
        base_model_name="meta-llama/Llama-3.1-8B"
    )
    
    prompt = "Write a contemplative passage about memory and loss."
    
    print(f"\nPrompt: {prompt}\n")
    print("Sampling...\n")
    
    response = await client.sample(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.8
    )
    
    print("="*60)
    print("GENERATED PROSE:")
    print("="*60)
    print(response.content)
    print("="*60)


if __name__ == "__main__":
    asyncio.run(sample())

