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
    
    # Get tokenizer (no renderer needed - we're using plain text)
    tokenizer = get_tokenizer(base_model)
    
    # Try multiple prompts that match training data styles
    prompts = [
        "Write a brief, lyrical passage using sparse language to capture a fleeting memory.\n\n",
        "Compose a short philosophical reflection on the weight of loss, using concrete imagery.\n\n",
        "Craft a minimalist passage about longing, with simple but resonant language.\n\n",
        "Write a contemplative moment about childhood, told through fragmented sensory details.\n\n",
        "Describe an intimate exchange between two people using understated emotional weight.\n\n"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"PROMPT {i}/{len(prompts)}:")
        print(f"{'='*60}")
        print(prompt.strip())
        print(f"{'='*60}\n")
        
        # Tokenize the prompt (with BOS token, matching training format)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)
        
        # Sample from the model with lower temperature for better quality
        sampling_params = tinker.SamplingParams(
            temperature=0.7,
            max_tokens=200,
            stop=[tokenizer.eos_token_id] if tokenizer.eos_token_id else []
        )
        
        response = await sampling_client.sample_async(
            prompt=prompt_input,
            sampling_params=sampling_params,
            num_samples=1
        )
        
        # Extract and decode the tokens
        sampled_tokens = response.sequences[0].tokens
        prose = tokenizer.decode(sampled_tokens, skip_special_tokens=True)
        
        print("GENERATED PROSE:")
        print("-"*60)
        print(prose)
        print("-"*60)


if __name__ == "__main__":
    asyncio.run(sample())

