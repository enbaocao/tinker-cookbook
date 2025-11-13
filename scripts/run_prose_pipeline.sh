#!/bin/bash
# Complete prose training pipeline runner
# Usage: ./run_prose_pipeline.sh prose_examples.md

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ "$#" -ne 1 ]; then
    echo -e "${RED}Usage: $0 <prose_markdown_file>${NC}"
    echo ""
    echo "Example:"
    echo "  $0 my_prose_examples.md"
    exit 1
fi

INPUT_MD="$1"

if [ ! -f "$INPUT_MD" ]; then
    echo -e "${RED}Error: File '$INPUT_MD' not found${NC}"
    exit 1
fi

# Check for required environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}Error: ANTHROPIC_API_KEY not set${NC}"
    echo ""
    echo "Set it with:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

# Derived filenames
BASENAME=$(basename "$INPUT_MD" .md)
PARSED_JSON="${BASENAME}_parsed.json"
TRAINING_JSONL="${BASENAME}_training.jsonl"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Prose Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Input:  ${YELLOW}$INPUT_MD${NC}"
echo -e "Parsed: ${YELLOW}$PARSED_JSON${NC}"
echo -e "Output: ${YELLOW}$TRAINING_JSONL${NC}"
echo ""

# Step 1: Parse markdown
echo -e "${GREEN}[1/2] Parsing markdown examples...${NC}"
python scripts/parse_prose_examples.py "$INPUT_MD" "$PARSED_JSON"

if [ ! -f "$PARSED_JSON" ]; then
    echo -e "${RED}Error: Parsing failed${NC}"
    exit 1
fi

echo ""

# Step 2: Label with Claude
echo -e "${GREEN}[2/2] Generating prompts with Claude Haiku...${NC}"
python scripts/label_with_claude.py "$PARSED_JSON" "$TRAINING_JSONL" \
    --batch-size 20 \
    --delay 0.5

if [ ! -f "$TRAINING_JSONL" ]; then
    echo -e "${RED}Error: Labeling failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Training data ready: ${YELLOW}$TRAINING_JSONL${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Inspect the output:"
echo "   head -3 $TRAINING_JSONL | python -m json.tool"
echo ""
echo "2. Train your LoRA:"
echo "   python train_prose_lora.py \\"
echo "     file_path=$TRAINING_JSONL \\"
echo "     model_name=meta-llama/Llama-3.1-8B"
echo ""
