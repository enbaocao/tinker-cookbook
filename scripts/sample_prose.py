#!/usr/bin/env python3
"""
Sample from a trained prose LoRA model.

Usage:
    python scripts/sample_prose.py
"""

import asyncio
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Your checkpoint from training (updated with simple training)
model_path = "tinker://1996f52b-06bd-5d23-8d20-7d7eb9fbfe73:train:0/sampler_weights/final"
base_model = "meta-llama/Llama-3.1-8B"


async def sample():
    print("Initializing sampling client...")
    
    # Create service client
    service_client = tinker.ServiceClient()
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(
        model_path=model_path,
        base_model=base_model
    )
    
    # Get tokenizer and renderer
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    # Try multiple prompts that match training data styles
    prompts = [
        "Write a brief, lyrical passage using sparse language to capture a fleeting memory.",
        "Compose a short philosophical reflection on the weight of loss, using concrete imagery.",
        "Craft a minimalist passage about longing, with simple but resonant language.",
        "Write a contemplative moment about childhood, told through fragmented sensory details.",
        "Describe an intimate exchange between two people using understated emotional weight."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"PROMPT {i}/{len(prompts)}:")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}\n")
        
        # Render the messages
        messages = [{"role": "user", "content": prompt}]
        prompt_text = renderer.build_generation_prompt(messages)
        
        # Sample from the model with lower temperature for better quality
        sampling_params = tinker.SamplingParams(
            temperature=0.6,  # Lower than before for more focused output
            max_tokens=150,
            stop=renderer.get_stop_sequences()
        )
        
        response = await sampling_client.sample_async(
            prompt=prompt_text,
            sampling_params=sampling_params,
            num_samples=1
        )
        
        # Extract the tokens from the first sample
        sampled_tokens = response.sequences[0].tokens
        
        # Parse the response
        parsed = renderer.parse_response(sampled_tokens)
        
        print("GENERATED PROSE:")
        print("-"*60)
        print(parsed[0]["content"] if isinstance(parsed, list) else parsed)
        print("-"*60)


if __name__ == "__main__":
    asyncio.run(sample())

