#!/bin/bash
#
# Run Baseline Comparison between PyTorch and Our Framework for LLaMA3 (No LoRA)
#
# This script verifies that the base LLaMA3 model (without LoRA) produces matching
# results between PyTorch and our framework.
#
# Usage:
#   bash tools/lora/run_baseline_comparison_llama3.sh
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

# Output directories
PYTORCH_OUTPUT_DIR="$PROJECT_ROOT/data/lora/pytorch_llama3_baseline"
OUR_OUTPUT_DIR="$PROJECT_ROOT/data/lora/our_llama3_baseline"

echo "=========================================="
echo "  Baseline Comparison (LLaMA3.2, No LoRA)"
echo "  PyTorch vs Our Framework"
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
echo "  LoRA:             DISABLED"
echo ""

# ============================================================================
# Step 1: Check Dependencies
# ============================================================================
echo "[Step 1/4] Checking dependencies..."

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
# Step 2: Run PyTorch Training (No LoRA)
# ============================================================================
echo "[Step 2/4] Running PyTorch training (no LoRA)..."
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
    --output_dir "$PYTORCH_OUTPUT_DIR"

echo "  PyTorch logs: $PYTORCH_OUTPUT_DIR/main.log"
echo ""

# ============================================================================
# Step 3: Run Our Framework Training (No LoRA)
# ============================================================================
echo "[Step 3/4] Running Our Framework training (no LoRA)..."
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
    --lora_rank 0 2>&1 | tee "$OUR_OUTPUT_DIR/main.log"

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
echo "  Baseline Comparison Results (LLaMA3.2)"
echo "=========================================="
echo ""

# Run compare_loss.py with tighter threshold for baseline
# Baseline should have much smaller differences than LoRA
python3 "$PROJECT_ROOT/scripts/compare_loss.py" "$PYTORCH_OUTPUT_DIR" "$OUR_OUTPUT_DIR" --threshold-fp32 1e-5 --verbose

echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
echo "PyTorch logs:  $PYTORCH_OUTPUT_DIR/main.log"
echo "Our logs:      $OUR_OUTPUT_DIR/main.log"
echo ""
