#!/bin/bash
#
# Run LoRA Loss Comparison between PyTorch and Our Framework
#
# This script:
# 1. Runs PyTorch GPT2 LoRA training
# 2. Runs Our Framework GPT2 LoRA training
# 3. Compares model architectures
# 4. Compares the losses
#
# Usage:
#   bash tools/lora/run_lora_comparison.sh
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
DATA_DIR="/data/shared/InfiniTrain-dev/data/llmc/gpt2"

# Training parameters
BATCH_SIZE=4
SEQUENCE_LENGTH=64
TOTAL_BATCH_SIZE=256
NUM_ITERATIONS=10
LEARNING_RATE=1e-5
DTYPE="float32"
DEVICE="cuda"
MODEL="gpt2"  # Use pretrained gpt2

# LoRA parameters (same for both frameworks)
LORA_RANK=8
LORA_ALPHA=16.0
LORA_TARGET_MODULES="c_attn,attn.c_proj"  # Only inject attention layers

# Output directories
PYTORCH_OUTPUT_DIR="$PROJECT_ROOT/data/lora/pytorch_logs"
OUR_OUTPUT_DIR="$PROJECT_ROOT/data/lora/our_logs"

echo "=========================================="
echo "  LoRA Loss Comparison"
echo "  PyTorch (PEFT) vs Our Framework"
echo "=========================================="
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
echo "LoRA configuration:"
echo "  Rank:             $LORA_RANK"
echo "  Alpha:            $LORA_ALPHA"
echo "  Target modules:   $LORA_TARGET_MODULES"
echo ""

# ============================================================================
# Step 1: Check Dependencies
# ============================================================================
echo "[Step 1/5] Checking dependencies..."

# Check PEFT library
if ! python3 -c "import peft" 2>/dev/null; then
    echo "  ERROR: PEFT library not installed"
    echo "  Install with: pip install peft"
    exit 1
fi
echo "  PEFT library: OK"

# Check our framework binary
if [ ! -f "$BUILD_DIR/gpt2" ]; then
    echo "  ERROR: Our framework binary not found at $BUILD_DIR/gpt2"
    echo "  Build with: cd $BUILD_DIR && cmake .. && make gpt2"
    exit 1
fi
echo "  Our framework binary: OK"

# Check data
if [ ! -f "$DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin" ]; then
    echo "  ERROR: Training data not found at $DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin"
    exit 1
fi
echo "  Training data: OK"
echo ""

# ============================================================================
# Step 2: Run PyTorch LoRA Training
# ============================================================================
echo "[Step 2/5] Running PyTorch (PEFT) LoRA training..."
mkdir -p "$PYTORCH_OUTPUT_DIR"

cd "$PROJECT_ROOT"

python3 pytorch_scripts/train_gpt2.py \
    --input_bin "$DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin" \
    --model "$MODEL" \
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
echo "[Step 3/5] Running Our Framework LoRA training..."
mkdir -p "$OUR_OUTPUT_DIR"

# Set GLOG to write logs to our output directory
# export GLOG_log_dir="$OUR_OUTPUT_DIR"
# export GLOG_v=0
# export GLOG_stderrthreshold=3  # Only ERROR and above go to stderr

"$BUILD_DIR/gpt2" \
    --device "$DEVICE" \
    --input_bin "$DATA_DIR/tinyshakespeare/tiny_shakespeare_train.bin" \
    --llmc_filepath "$DATA_DIR/gpt2_124M.bin" \
    --batch_size "$BATCH_SIZE" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --total_batch_size "$TOTAL_BATCH_SIZE" \
    --num_iteration "$NUM_ITERATIONS" \
    --learning_rate "$LEARNING_RATE" \
    --dtype "$DTYPE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules "$LORA_TARGET_MODULES" 2>&1 | tee "$OUR_OUTPUT_DIR/main.log"

# Merge GLOG files into main.log (INFO logs go to separate files)
# GLOG creates files like: gpt2.node24.root.log.INFO.20260211-075430.844114
for logfile in "$OUR_OUTPUT_DIR"/gpt2.*.log.*; do
    if [ -f "$logfile" ] 2>/dev/null; then
        cat "$logfile" >> "$OUR_OUTPUT_DIR/main.log"
        rm -f "$logfile"
    fi
done

# Also merge symlinks
for symlink in "$OUR_OUTPUT_DIR"/gpt2.*; do
    if [ -L "$symlink" ] 2>/dev/null; then
        rm -f "$symlink"
    fi
done

echo "  Our framework logs: $OUR_OUTPUT_DIR/main.log"
echo ""

# ============================================================================
# Step 4: Compare Parameter Names
# ============================================================================
echo "[Step 4/5] Comparing parameter names..."

echo ""
echo "===== Extracting parameter names from PyTorch logs ====="
PYTORCH_PARAMS="$PYTORCH_OUTPUT_DIR/params_pytorch.txt"
grep -E "^\s*.*lora_[AB]" "$PYTORCH_OUTPUT_DIR/main.log" > "$PYTORCH_PARAMS" 2>/dev/null || true
echo "Found $(wc -l < "$PYTORCH_PARAMS") trainable parameters in PyTorch"

echo ""
echo "===== Extracting parameter names from Our Framework logs ====="
OUR_PARAMS="$OUR_OUTPUT_DIR/params_our.txt"
grep -E "lora_[AB]" "$OUR_OUTPUT_DIR/main.log" > "$OUR_PARAMS" 2>/dev/null || true
echo "Found $(wc -l < "$OUR_PARAMS") parameters in our framework"

echo ""
echo "===== Parameter comparison saved to ====="
echo "  PyTorch: $PYTORCH_PARAMS"
echo "  Ours:   $OUR_PARAMS"

# ============================================================================
# Step 5: Compare Losses
# ============================================================================
echo "[Step 5/5] Comparing losses..."

echo ""
echo "=========================================="
echo "  Comparison Results"
echo "=========================================="
echo ""

# Run compare_loss.py
# Note: Cross-framework loss comparison uses threshold 1e-3 due to:
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
