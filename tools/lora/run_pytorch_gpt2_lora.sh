#!/bin/bash
#
# PyTorch GPT2 LoRA Training Script
#
# This script runs GPT2 LoRA training using PEFT library
# and compares the loss with our framework's implementation.
#
# Usage:
#   bash tools/lora/run_pytorch_gpt2_lora.sh
#

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="/data/shared/InfiniTrain-dev/data/llmc/gpt2"

# Model and data
MODEL="gpt2"  # gpt2, gpt2-medium, gpt2-large, gpt2-xl
INPUT_BIN="$DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin"

# Training parameters
BATCH_SIZE=4
SEQUENCE_LENGTH=64
TOTAL_BATCH_SIZE=256
NUM_ITERATIONS=10
LEARNING_RATE=1e-5
DTYPE="float32"

# LoRA parameters
LORA_RANK=8
LORA_ALPHA=16.0
LORA_TARGET_MODULES="c_attn,attn.c_proj"  # Only inject attention layers

# Output
OUTPUT_DIR="$PROJECT_ROOT/data/lora/pytorch_logs"

echo "=========================================="
echo "  PyTorch GPT2 LoRA Training"
echo "=========================================="
echo ""
echo "Data dir:        $DATA_DIR"
echo "Output dir:      $OUTPUT_DIR"
echo ""
echo "Training configuration:"
echo "  Model:            $MODEL"
echo "  Batch size:       $BATCH_SIZE"
echo "  Sequence length:  $SEQUENCE_LENGTH"
echo "  Total batch size: $TOTAL_BATCH_SIZE"
echo "  Iterations:       $NUM_ITERATIONS"
echo "  Learning rate:    $LEARNING_RATE"
echo "  Data type:        $DTYPE"
echo ""
echo "LoRA configuration (PEFT):"
echo "  Rank:             $LORA_RANK"
echo "  Alpha:            $LORA_ALPHA"
echo "  Target modules:   $LORA_TARGET_MODULES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run PyTorch LoRA training
cd "$PROJECT_ROOT"

python pytorch_train_gpt2.py \
    --input_bin "$INPUT_BIN" \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --total_batch_size "$TOTAL_BATCH_SIZE" \
    --num_iterations "$NUM_ITERATIONS" \
    --learning_rate "$LEARNING_RATE" \
    --dtype "$DTYPE" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --lora_save_path "$OUTPUT_DIR/lora_weights"

echo ""
echo "=========================================="
echo "  PyTorch LoRA Training Complete!"
echo "=========================================="
echo ""
echo "Logs saved to: $OUTPUT_DIR/main.log"
echo ""

# ============================================================================
# Compare with our framework
# ============================================================================
echo "=========================================="
echo "  Loss Comparison Setup"
echo "=========================================="
echo ""
echo "To compare with our framework, run:"
echo ""
echo "1. Run our framework's LoRA training:"
echo "   ./build/gpt2 --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA ..."
echo ""
echo "2. Compare losses:"
echo "   python scripts/compare_loss.py $OUTPUT_DIR /path/to/our_logs --threshold-fp32 1e-5"
echo ""
