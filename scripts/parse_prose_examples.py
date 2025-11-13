#!/usr/bin/env python3
"""
Parse a markdown file containing prose examples into structured JSON.

Expected markdown format:
    ##### title (author)
    prose paragraph 1

    prose paragraph 2

    ##### next title (author)
    prose paragraph 1
    ...

Usage:
    python parse_prose_examples.py input.md output.json
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict


def parse_prose_markdown(markdown_path: str) -> List[Dict[str, str]]:
    """
    Parse markdown file with prose examples.

    Returns list of dicts with keys: title, author, prose
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()

    examples = []

    # Split by h5 headers (##### )
    # Pattern: ##### title (author)
    sections = re.split(r'^#{5}\s+', content, flags=re.MULTILINE)

    # First section before any header is typically empty or intro text
    sections = [s.strip() for s in sections if s.strip()]

    for section in sections:
        # Extract title and author from first line
        lines = section.split('\n', 1)
        if len(lines) < 2:
            continue

        header = lines[0].strip()
        prose_content = lines[1].strip() if len(lines) > 1 else ""

        # Parse header: "title (author)"
        match = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', header)
        if match:
            title = match.group(1).strip()
            author = match.group(2).strip()
        else:
            # Fallback if format doesn't match
            title = header
            author = "Unknown"

        # Clean up prose: remove excessive whitespace but preserve paragraph breaks
        prose_lines = [line.strip() for line in prose_content.split('\n')]
        prose_paragraphs = []

        current_para = []
        for line in prose_lines:
            if line:
                current_para.append(line)
            elif current_para:
                prose_paragraphs.append(' '.join(current_para))
                current_para = []

        # Don't forget last paragraph
        if current_para:
            prose_paragraphs.append(' '.join(current_para))

        # Join paragraphs with double newline
        prose = '\n\n'.join(prose_paragraphs)

        if prose:  # Only add if there's actual prose content
            examples.append({
                'title': title,
                'author': author,
                'prose': prose,
                'num_paragraphs': len(prose_paragraphs),
                'char_count': len(prose)
            })

    return examples


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_prose_examples.py <input.md> <output.json>")
        print("\nExample:")
        print("  python parse_prose_examples.py prose_examples.md parsed_examples.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(input_path).exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)

    print(f"Parsing {input_path}...")
    examples = parse_prose_markdown(input_path)

    print(f"\nParsed {len(examples)} prose examples")
    print(f"Average length: {sum(e['char_count'] for e in examples) // len(examples)} characters")

    # Show first few examples
    print(f"\nFirst 3 examples:")
    for i, example in enumerate(examples[:3], 1):
        print(f"\n{i}. {example['title']} by {example['author']}")
        print(f"   Paragraphs: {example['num_paragraphs']}, Chars: {example['char_count']}")
        preview = example['prose'][:100] + "..." if len(example['prose']) > 100 else example['prose']
        print(f"   Preview: {preview}")

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(examples)} examples to {output_path}")
    print(f"\nNext step:")
    print(f"  python label_with_claude.py {output_path} training_data.jsonl")


if __name__ == "__main__":
    main()
