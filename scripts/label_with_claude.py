#!/usr/bin/env python3
"""
Use Claude API to generate creative prompts for prose examples.
Creates training-ready JSONL for tinker-cookbook.

Usage:
    export ANTHROPIC_API_KEY=your_api_key
    python label_with_claude.py parsed_examples.json output.jsonl

Optional arguments:
    --model MODEL          Claude model to use (default: claude-haiku-4.5-20250702)
    --resume              Resume from existing output file
    --batch-size N        Process N examples before saving (default: 10)
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)


PROMPT_GENERATION_SYSTEM = """You are an expert creative writing instructor designing prompts for an AI writing assistant.

Your task: Generate a concise, evocative prompt that would naturally elicit the given prose example.

Guidelines:
- The prompt should be 1-2 sentences
- Focus on style, tone, theme, or scenario
- Be specific enough to guide style but open-ended enough for creativity
- Vary your prompts: use different structures like:
  * "Write a [genre] passage about [topic] focusing on [element]"
  * "Describe [scene] in a [style] voice"
  * "Craft a [length] piece exploring [theme]"
  * "Compose a paragraph that [goal/feeling]"
- Match the sophistication level of the prose
- Don't mention the author or source

Examples:
- For lyrical prose: "Write a contemplative passage exploring the relationship between memory and place, using vivid sensory details."
- For sharp social commentary: "Craft a brief, incisive observation about power dynamics in modern society."
- For poetic description: "Describe a natural phenomenon using metaphorical language and rich imagery."

Return ONLY the prompt text, nothing else."""


def generate_prompt_for_prose(
    client: anthropic.Anthropic,
    prose: str,
    title: str,
    author: str,
    model: str = "claude-haiku-4.5-20250702"
) -> str:
    """
    Use Claude to generate a creative prompt for the given prose.
    """
    user_message = f"""Generate a writing prompt that would elicit prose similar to this example:

Title: {title}
Author: {author}

Prose:
{prose}

Remember: Output ONLY the prompt, no explanations or quotes."""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.8,  # Some creativity in prompt generation
            system=PROMPT_GENERATION_SYSTEM,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        prompt = message.content[0].text.strip()

        # Remove quotes if Claude added them
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1]
        if prompt.startswith("'") and prompt.endswith("'"):
            prompt = prompt[1:-1]

        return prompt

    except anthropic.APIError as e:
        print(f"\n⚠ API Error: {e}")
        raise


def load_existing_output(output_path: str) -> Dict[str, Dict]:
    """
    Load existing JSONL output to support resume functionality.
    Returns dict mapping prose -> entry for deduplication.
    """
    if not Path(output_path).exists():
        return {}

    existing = {}
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Use prose content as key for deduplication
                prose = entry['messages'][1]['content']
                existing[prose] = entry

    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Label prose examples with Claude-generated prompts"
    )
    parser.add_argument('input_json', help='Input JSON file from parse_prose_examples.py')
    parser.add_argument('output_jsonl', help='Output JSONL file for training')
    parser.add_argument('--model', default='claude-haiku-4.5-20250702',
                       help='Claude model to use')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing output file')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Save progress every N examples')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API calls in seconds')

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export ANTHROPIC_API_KEY=your_api_key")
        sys.exit(1)

    # Load input
    if not Path(args.input_json).exists():
        print(f"Error: Input file '{args.input_json}' not found")
        sys.exit(1)

    with open(args.input_json, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} prose examples from {args.input_json}")

    # Load existing output if resuming
    existing = {}
    if args.resume:
        existing = load_existing_output(args.output_jsonl)
        print(f"Resuming: Found {len(existing)} existing labeled examples")

    # Filter out already processed
    to_process = [ex for ex in examples if ex['prose'] not in existing]

    if not to_process:
        print("\n✓ All examples already processed!")
        print(f"Output: {args.output_jsonl}")
        sys.exit(0)

    print(f"Will process {len(to_process)} new examples")
    print(f"Using model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)

    # Open output file in append mode
    mode = 'a' if args.resume else 'w'
    output_file = open(args.output_jsonl, mode, encoding='utf-8')

    successful = 0
    failed = 0

    try:
        for i, example in enumerate(to_process, 1):
            print(f"[{i}/{len(to_process)}] Processing: {example['title']} by {example['author']}")

            try:
                # Generate prompt using Claude
                prompt = generate_prompt_for_prose(
                    client=client,
                    prose=example['prose'],
                    title=example['title'],
                    author=example['author'],
                    model=args.model
                )

                # Create training format
                training_entry = {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "assistant",
                            "content": example['prose']
                        }
                    ],
                    "metadata": {
                        "title": example['title'],
                        "author": example['author'],
                        "char_count": example['char_count']
                    }
                }

                # Write to JSONL
                output_file.write(json.dumps(training_entry, ensure_ascii=False) + '\n')

                successful += 1

                # Show generated prompt
                print(f"  → Prompt: {prompt[:80]}...")

                # Periodic flush
                if i % args.batch_size == 0:
                    output_file.flush()
                    print(f"\n✓ Progress saved ({successful} examples)\n")

                # Rate limiting
                if i < len(to_process):  # Don't delay after last item
                    time.sleep(args.delay)

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed += 1
                continue

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    finally:
        output_file.close()

    # Summary
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"Successfully labeled: {successful}")
    print(f"Failed: {failed}")
    print(f"Total in output file: {successful + len(existing)}")
    print(f"\nOutput: {args.output_jsonl}")

    if failed > 0:
        print(f"\n⚠ {failed} examples failed. Run with --resume to retry.")

    print("\n✓ Ready for training!")
    print(f"\nNext step:")
    print(f"  python -m tinker_cookbook.recipes.sl_basic \\")
    print(f"    dataset_path={args.output_jsonl} \\")
    print(f"    model_name=meta-llama/Llama-3.1-8B")


if __name__ == "__main__":
    main()
