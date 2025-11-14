#!/usr/bin/env python3
"""
Sample from the base Llama 3.1 8B model (no LoRA) for comparison.

Usage:
    python scripts/sample_base_model.py
"""

import asyncio
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


base_model = "meta-llama/Llama-3.1-8B"


async def sample():
    print("Initializing base model sampling client...")
    
    # Create service client
    service_client = tinker.ServiceClient()
    
    # Create sampling client WITHOUT model_path (uses base model only)
    sampling_client = service_client.create_sampling_client(
        base_model=base_model
    )
    
    # Get tokenizer and renderer for proper chat formatting
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    # Same prompts as the LoRA test (without \n\n since renderer handles formatting)
    prompts = [
        "Write a brief, lyrical passage using sparse language to capture a fleeting memory.",
        "Compose a short philosophical reflection on the weight of loss, using concrete imagery.",
        "Craft a minimalist passage about longing, with simple but resonant language.",
        "Write a contemplative moment about childhood, told through fragmented sensory details.",
        "Describe an intimate exchange between two people using understated emotional weight."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"PROMPT {i}/{len(prompts)} (BASE MODEL WITH CHAT FORMAT):")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}\n")
        
        # Use renderer to format as proper chat
        messages = [{"role": "user", "content": prompt}]
        prompt_text = renderer.build_generation_prompt(messages)
        
        # Sample from the model
        sampling_params = tinker.SamplingParams(
            temperature=0.7,
            max_tokens=200,
            stop=renderer.get_stop_sequences()
        )
        
        response = await sampling_client.sample_async(
            prompt=prompt_text,
            sampling_params=sampling_params,
            num_samples=1
        )
        
        # Extract and parse the tokens
        sampled_tokens = response.sequences[0].tokens
        parsed = renderer.parse_response(sampled_tokens)
        
        print("GENERATED PROSE (BASE MODEL):")
        print("-"*60)
        print(parsed[0]["content"] if isinstance(parsed, list) else parsed)
        print("-"*60)


if __name__ == "__main__":
    asyncio.run(sample())

