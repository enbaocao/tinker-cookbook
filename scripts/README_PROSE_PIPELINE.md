# Prose Training Data Pipeline

This directory contains scripts to prepare prose examples for training a LoRA writing model.

## Overview

**Pipeline Flow:**
```
prose_examples.md → parse_prose_examples.py → parsed.json → label_with_claude.py → training_data.jsonl → tinker training
```

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install anthropic

# Set up API key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export TINKER_API_KEY=your_tinker_api_key
```

### Step 1: Prepare Your Markdown File

Format your prose examples as:

```markdown
##### title (author)
First prose example from this source.

Second prose example from the same source.

Third example also from this source.

##### another title (different author)
A prose example from a different source.

Another example from this second source.
```

**Requirements:**
- Each section starts with `##### title (author)`
- **Each paragraph (separated by blank lines) becomes its own training example**
- All paragraphs under one header share the same title/author metadata
- One header can contain many prose examples (one per paragraph)

### Step 2: Parse Examples

Extract structured data from your markdown:

```bash
python scripts/parse_prose_examples.py prose_examples.md parsed_examples.json
```

**Output:** `parsed_examples.json` - structured list of examples with metadata

**Inspect the output:**
```bash
# Check parsing quality
python -c "import json; d=json.load(open('parsed_examples.json')); print(f'{len(d)} examples'); print(json.dumps(d[0], indent=2))"
```

### Step 3: Generate Prompts with Claude

Use Claude Haiku 4.5 to generate creative writing prompts:

```bash
python scripts/label_with_claude.py parsed_examples.json training_data.jsonl
```

**Options:**
```bash
# Use different model
python scripts/label_with_claude.py parsed_examples.json training_data.jsonl \
  --model claude-opus-4-20250514

# Resume interrupted labeling
python scripts/label_with_claude.py parsed_examples.json training_data.jsonl \
  --resume

# Adjust rate limiting
python scripts/label_with_claude.py parsed_examples.json training_data.jsonl \
  --delay 1.0 \
  --batch-size 20
```

**Output:** `training_data.jsonl` - ready for tinker training

### Step 4: Train Your LoRA

Create a training script (`train_prose_lora.py`):

```python
import chz
import sys
import asyncio
from tinker_cookbook import cli_utils, hyperparam_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B"
    learning_rate = hyperparam_utils.get_lr(model_name, is_lora=True)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name="llama3",
        max_length=2048,
        batch_size=64,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path="training_data.jsonl",
        test_size=50,
        shuffle_seed=42,
    )

    return chz.Blueprint(train.Config).apply({
        "log_path": "/tmp/tinker-prose-lora",
        "model_name": model_name,
        "dataset_builder": dataset,
        "learning_rate": learning_rate,
        "lora_rank": 32,
        "lr_schedule": "linear",
        "num_epochs": 3,
        "eval_every": 16,
        "save_every": 100,
    })

def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))

if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
```

Run training:
```bash
python train_prose_lora.py
```

## Script Details

### `parse_prose_examples.py`

**Purpose:** Parse markdown into structured JSON

**Input format:**
```markdown
##### comforting myths (rabih alameddine)
the writers who are allowed to talk are those who prop up the dominant culture, who reflect it with a gilded mirror.

We invade your countries, destroy your economies, demolish your infrastructures, murder hundreds of thousands of your citizens, and a decade or so later we write beautifully restrained novels about how killing you made us cry.

opposing the dominant culture is like trying to whittle down a mountain by rubbing it with a silk scarf. Yet a writer must. I may not be able to move mountains like Superman, but I have lovely scarves.
```

**Output format:**
```json
[
  {
    "title": "comforting myths",
    "author": "rabih alameddine",
    "prose": "the writers who are allowed to talk are those who prop up the dominant culture, who reflect it with a gilded mirror.",
    "char_count": 114
  },
  {
    "title": "comforting myths",
    "author": "rabih alameddine",
    "prose": "We invade your countries, destroy your economies, demolish your infrastructures...",
    "char_count": 178
  },
  {
    "title": "obituary for dead languages",
    "author": "heather altfeld",
    "prose": "Here are the continents, once married, now divorced by the currents of the sea...",
    "char_count": 156
  }
]
```

Note: Each paragraph becomes its own entry with the same title/author.

### `label_with_claude.py`

**Purpose:** Generate creative prompts using Claude API

**Features:**
- Uses Claude Haiku 4.5 (cost-effective)
- Generates diverse prompt styles
- Auto-saves progress every 10 examples
- Supports resume after interruption
- Rate limiting to avoid API throttling

**Prompt Generation System:**
The script instructs Claude to create varied prompts like:
- "Write a lyrical passage exploring themes of exile and belonging"
- "Craft a sharp observation about cultural power dynamics"
- "Compose a meditative paragraph on the intersection of violence and memory"

**Output format (JSONL):**
```json
{"messages": [{"role": "user", "content": "Write a contemplative passage..."}, {"role": "assistant", "content": "the writers who are allowed..."}], "metadata": {"title": "comforting myths", "author": "rabih alameddine"}}
```

## Tips & Best Practices

### For Parsing

✅ **Do:**
- Keep consistent header format: `##### title (author)`
- Use blank lines between paragraphs
- Check parsed output before labeling

❌ **Avoid:**
- Mixing header formats (e.g., `####` vs `#####`)
- Including non-prose content (like notes/comments)
- Extra markup within prose (italics are ok)

### For Labeling

✅ **Do:**
- Use `--resume` if interrupted
- Check first few generated prompts for quality
- Adjust `--delay` if hitting rate limits
- Keep `--batch-size` at 10-20 for safe progress saves

❌ **Avoid:**
- Running without rate limiting (may hit API limits)
- Processing without inspecting parsed data first
- Forgetting to set `ANTHROPIC_API_KEY`

### For Training

✅ **Do:**
- Use `TrainOnWhat.ALL_ASSISTANT_MESSAGES` (trains on prose only)
- Set test_size to 50-100 examples for evaluation
- Use `hyperparam_utils.get_lr()` for optimal learning rate
- Start with LoRA rank 32, increase to 64 if needed
- Monitor eval loss to detect overfitting

❌ **Avoid:**
- Training on prompts (`ALL_MESSAGES` trains on both)
- Too low learning rate (LoRA needs ~10x higher than full fine-tuning)
- Skipping test set evaluation
- Training for too many epochs (watch for overfitting)

## Cost Estimation

**Claude API (Haiku 4.5):**
- Input: ~$0.10 per 1M tokens
- Output: ~$0.50 per 1M tokens
- Per example: ~200 input + 50 output tokens
- **500 examples ≈ $0.03**

**Tinker Training:**
- Varies by model size and training time
- See Tinker pricing: https://tinker-docs.thinkingmachines.ai/

## Troubleshooting

**"ANTHROPIC_API_KEY not set"**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**"Failed to generate prompt"**
- Check API key is valid
- Increase `--delay` to avoid rate limits
- Use `--resume` to skip already processed

**"Output JSONL has wrong format"**
- Inspect with: `head -1 training_data.jsonl | python -m json.tool`
- Should have `messages` array with `role` and `content`

**Training loss not decreasing:**
- Check learning rate (use `hyperparam_utils.get_lr()`)
- Ensure using `ALL_ASSISTANT_MESSAGES` not `ALL_MESSAGES`
- Verify data quality (no corrupted examples)
- Try increasing LoRA rank to 64

## Example: Full Pipeline

```bash
# 1. Parse your markdown
python scripts/parse_prose_examples.py my_prose.md parsed.json

# Output: Parsed 487 prose examples

# 2. Generate prompts with Claude
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/label_with_claude.py parsed.json training.jsonl --batch-size 20

# Output: Successfully labeled: 487

# 3. Inspect the output
head -3 training.jsonl | python -m json.tool

# 4. Train the LoRA
python train_prose_lora.py \
  file_path=training.jsonl \
  model_name=meta-llama/Llama-3.1-8B \
  num_epochs=3

# 5. Test your model
# (After training completes, use sampling client to generate prose)
```

## Advanced: Custom Prompt Templates

Edit `label_with_claude.py` to customize prompt generation:

```python
# Around line 30, modify PROMPT_GENERATION_SYSTEM
PROMPT_GENERATION_SYSTEM = """Your custom instructions here...

Example prompt formats:
- "In the style of [author], write about [topic]"
- "Create a [length] piece with [characteristics]"
"""
```

## Questions?

- Tinker docs: https://tinker-docs.thinkingmachines.ai/
- Anthropic API: https://docs.anthropic.com/
- Issues: https://github.com/anthropics/tinker-cookbook/issues
