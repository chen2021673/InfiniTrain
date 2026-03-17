#!/bin/bash
#
# Run LoRA Loss Comparison between PyTorch and Our Framework for LLaMA3.2
#
# This script:
# 1. Runs PyTorch LLaMA3.2 LoRA training
# 2. Runs Our Framework LLaMA3.2 LoRA training
# 3. Compares the losses
#
# Usage:
#   bash tools/lora/run_lora_comparison_llama3.sh
#

set -e
source /opt/miniconda3/bin/activate pytorch_env
# A800模型目录
export HF_HOME=/data/shared/InfiniTrain-dev/env/HuggingFace

# 可选，直接设置模型缓存目录
export HF_HUB_CACHE=/data/shared/InfiniTrain-dev/env/HuggingFace/hub
# 可选，强制使用本地文件，不检查远程更新
export HF_HUB_OFFLINE=1
# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
DATA_DIR="/data/shared/InfiniTrain-dev/data/llmc/llama3"

# Training parameters
BATCH_SIZE=4
SEQUENCE_LENGTH=64
TOTAL_BATCH_SIZE=256
NUM_ITERATIONS=10
LEARNING_RATE=1e-5
DTYPE="float32"
DEVICE="cuda"

# LoRA parameters (same for both frameworks)
LORA_RANK=8
LORA_ALPHA=16.0
LORA_TARGET_MODULES="c_attn,attn.c_proj"  # Only inject attention layers

# Output directories
PYTORCH_OUTPUT_DIR="$PROJECT_ROOT/data/lora/pytorch_llama3_logs"
OUR_OUTPUT_DIR="$PROJECT_ROOT/data/lora/our_llama3_logs"

echo "=========================================="
echo "  LoRA Loss Comparison (LLaMA3.2)"
echo "  PyTorch (PEFT) vs Our Framework"
echo "=========================================="
echo ""
echo "Training configuration:"
echo "  Model:            LLaMA3.2-1B"
echo "  Batch size:       $BATCH_SIZE"
echo "  Sequence length: $SEQUENCE_LENGTH"
echo "  Total batch size: $TOTAL_BATCH_SIZE"
echo "  Iterations:       $NUM_ITERATIONS"
echo "  Learning rate:    $LEARNING_RATE"
echo "  Data type:        $DTYPE"
echo ""
echo "LoRA configuration:"
echo "  Rank:             $LORA_RANK"
echo "  Alpha:            $LORA_ALPHA"
echo "  Target modules:   $LORA_TARGET_MODULES"
echo ""

# ============================================================================
# Step 1: Check Dependencies
# ============================================================================
echo "[Step 1/4] Checking dependencies..."

# Check PEFT library
if ! python3 -c "import peft" 2>/dev/null; then
    echo "  ERROR: PEFT library not installed"
    echo "  Install with: pip install peft"
    exit 1
fi
echo "  PEFT library: OK"

# Check our framework binary
if [ ! -f "$BUILD_DIR/llama3" ]; then
    echo "  ERROR: Our framework binary not found at $BUILD_DIR/llama3"
    echo "  Build with: cd $BUILD_DIR && cmake .. && make llama3"
    exit 1
fi
echo "  Our framework binary: OK"

# Check data
if [ ! -f "$DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin" ]; then
    echo "  ERROR: Training data not found at $DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin"
    exit 1
fi
echo "  Training data: OK"

if [ ! -f "$DATA_DIR/llama3.2_1B_fp32.bin" ]; then
    echo "  ERROR: Model file not found at $DATA_DIR/llama3.2_1B_fp32.bin"
    exit 1
fi
echo "  Model file: OK"
echo ""

# ============================================================================
# Step 2: Run PyTorch LoRA Training
# ============================================================================
echo "[Step 2/4] Running PyTorch (PEFT) LoRA training..."
mkdir -p "$PYTORCH_OUTPUT_DIR"

cd "$PROJECT_ROOT"

python3 pytorch_scripts/train_llama3.2_1B.py \
    --input_bin "$DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin" \
    --batch_size "$BATCH_SIZE" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --total_batch_size "$TOTAL_BATCH_SIZE" \
    --num_iterations "$NUM_ITERATIONS" \
    --learning_rate "$LEARNING_RATE" \
    --dtype "$DTYPE" \
    --output_dir "$PYTORCH_OUTPUT_DIR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --lora_save_path "$PYTORCH_OUTPUT_DIR/lora_weights"

echo "  PyTorch logs: $PYTORCH_OUTPUT_DIR/main.log"
echo ""

# ============================================================================
# Step 3: Run Our Framework LoRA Training
# ============================================================================
echo "[Step 3/4] Running Our Framework LoRA training..."
mkdir -p "$OUR_OUTPUT_DIR"

"$BUILD_DIR/llama3" \
    --device "$DEVICE" \
    --input_bin "$DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin" \
    --llmc_filepath "$DATA_DIR/llama3.2_1B_fp32.bin" \
    --batch_size "$BATCH_SIZE" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --total_batch_size "$TOTAL_BATCH_SIZE" \
    --num_iteration "$NUM_ITERATIONS" \
    --learning_rate "$LEARNING_RATE" \
    --dtype "$DTYPE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --lora_load_path "$PYTORCH_OUTPUT_DIR/lora_weights" 2>&1 | tee "$OUR_OUTPUT_DIR/main.log"

# Merge GLOG files into main.log (INFO logs go to separate files)
for logfile in "$OUR_OUTPUT_DIR"/llama3.*.log.*; do
    if [ -f "$logfile" ] 2>/dev/null; then
        cat "$logfile" >> "$OUR_OUTPUT_DIR/main.log"
        rm -f "$logfile"
    fi
done

# Also merge symlinks
for symlink in "$OUR_OUTPUT_DIR"/llama3.*; do
    if [ -L "$symlink" ] 2>/dev/null; then
        rm -f "$symlink"
    fi
done

echo "  Our framework logs: $OUR_OUTPUT_DIR/main.log"
echo ""

# ============================================================================
# Step 4: Compare Losses
# ============================================================================
echo "[Step 4/4] Comparing losses..."

echo ""
echo "=========================================="
echo "  Comparison Results"
echo "=========================================="
echo ""

# Run compare_loss.py
# Note: Cross-framework loss comparison uses threshold 1e-5 due to:
# - Different random seeds in data shuffling
# - Different reduction order in gradient computation
# - Minor floating-point accumulation differences
python3 "$PROJECT_ROOT/scripts/compare_loss.py" "$PYTORCH_OUTPUT_DIR" "$OUR_OUTPUT_DIR" --threshold-fp32 1e-5 --verbose --plot

echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
echo "PyTorch logs:  $PYTORCH_OUTPUT_DIR/main.log"
echo "Our logs:      $OUR_OUTPUT_DIR/main.log"
echo ""
