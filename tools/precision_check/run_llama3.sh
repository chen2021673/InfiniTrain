#!/bin/bash
# InfiniTrain Precision Checker - Llama3
set -e

# Configuration
BIN="./build/llama3"
MODEL_ARGS="--device cuda --input_bin /data/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin --llmc_filepath /data/shared/InfiniTrain-dev/data/llmc/llama3/llama3.2_1B_fp32.bin"
OUTPUT_DIR="./log_precision_check_llama3"
COMPARE_SCRIPT="tools/precision_check/compare.py"

echo "=== InfiniTrain Precision Checker - Llama3 ==="

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Please build the project first."
    exit 1
fi

if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo "Error: $COMPARE_SCRIPT not found."
    exit 1
fi

# Clean test directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 1. Single-rank test - Simple format
echo ""
echo "=== 1. Single-rank test (Simple format) ==="
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test1_simple,format=simple,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test1_simple" | head -1)
echo "Timestamp directory: $TIMESTAMP_DIR"
NPY_COUNT=$(ls "$OUTPUT_DIR/test1_simple/$TIMESTAMP_DIR/rank_0"/*.npy 2>/dev/null | wc -l)
echo "Rank 0 NPY files: $NPY_COUNT"
LOG_FILE=$(ls "$OUTPUT_DIR/test1_simple/$TIMESTAMP_DIR"/*.log 2>/dev/null | head -1)
echo "Log file: $LOG_FILE"

# 2. Single-rank test - MD5 format
echo ""
echo "=== 2. Single-rank test (MD5 format) ==="
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test2_md5,format=md5\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test2_md5" | head -1)

# 3. Multi-iter overwrite test
echo ""
echo "=== 3. Multi-iter overwrite test ==="
CMD="$BIN $MODEL_ARGS --precision_check \"level=1,path=$OUTPUT_DIR/test3_overwrite,save_tensors=true\" --num_iteration 3"
echo "Running: $CMD"
eval $CMD

TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test3_overwrite" | head -1)
FILE_COUNT=$(ls "$OUTPUT_DIR/test3_overwrite/$TIMESTAMP_DIR/rank_0"/*.npy 2>/dev/null | wc -l)
echo "Files after 3 iters: $FILE_COUNT (should be same as 1 iter - files overwritten)"

# 4. Multi-rank test
echo ""
echo "=== 4. Multi-rank test ==="
CMD="$BIN $MODEL_ARGS --nthread_per_process 8 --tensor_parallel 4 --pipeline_parallel 2 --precision_check \"level=1,path=$OUTPUT_DIR/test4_multi,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

MULTI_TIMESTAMP_DIR=$(ls -t "$OUTPUT_DIR/test4_multi" | head -1)
MULTI_RUN_DIR="$OUTPUT_DIR/test4_multi/$MULTI_TIMESTAMP_DIR"

# 5. Comparison test (same-framework)
echo ""
echo "=== 5. Comparison test (same-framework) ==="
# Use multi-rank for comparison to test cross-rank comparison
CMD="$BIN $MODEL_ARGS --nthread_per_process 8 --tensor_parallel 4 --pipeline_parallel 2 --precision_check \"level=1,path=$OUTPUT_DIR/test5_compare_run1,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD
sleep 2
CMD="$BIN $MODEL_ARGS --nthread_per_process 8 --tensor_parallel 4 --pipeline_parallel 2 --precision_check \"level=1,path=$OUTPUT_DIR/test5_compare_run2,save_tensors=true\" --num_iteration 1"
echo "Running: $CMD"
eval $CMD

RUN1_DIR="$OUTPUT_DIR/test5_compare_run1/$(ls -t "$OUTPUT_DIR/test5_compare_run1" | head -1)"
RUN2_DIR="$OUTPUT_DIR/test5_compare_run2/$(ls -t "$OUTPUT_DIR/test5_compare_run2" | head -1)"

echo "Comparing two runs (rank_0):"
echo "  Run1: $RUN1_DIR/rank_0"
echo "  Run2: $RUN2_DIR/rank_0"
python "$COMPARE_SCRIPT" --dir1 "$RUN1_DIR/rank_0" --dir2 "$RUN2_DIR/rank_0" --atol 1e-5 --rtol 1e-3 || true

# Compare rank_1 between two runs
echo ""
echo "Comparing two runs (rank_1):"
if [ -d "$RUN1_DIR/rank_1" ] && [ -d "$RUN2_DIR/rank_1" ]; then
    echo "  Run1: $RUN1_DIR/rank_1"
    echo "  Run2: $RUN2_DIR/rank_1"
    python "$COMPARE_SCRIPT" --dir1 "$RUN1_DIR/rank_1" --dir2 "$RUN2_DIR/rank_1" --atol 1e-5 --rtol 1e-3 || true
else
    echo "Skipping: rank_1 output not available"
fi

# 6. Error detection test (verify compare.py can detect misaligned tensors)
echo ""
echo "=== 6. Error detection test ==="
# Create a corrupted copy of run1
CORRUPTED_DIR="$OUTPUT_DIR/test6_error"
mkdir -p "$CORRUPTED_DIR/rank_0"
cp "$RUN1_DIR/rank_0"/*.npy "$CORRUPTED_DIR/rank_0/" 2>/dev/null || true

# Corrupt one file by adding noise
python3 -c "
import numpy as np
import glob
files = glob.glob('$CORRUPTED_DIR/rank_0/*.npy')
if files:
    f = files[0]
    arr = np.load(f)
    arr = arr + np.random.randn(*arr.shape).astype(arr.dtype) * 0.1  # Add noise
    np.save(f, arr)
    print(f'Corrupted: {f}')
"

echo "Comparing run1 with corrupted copy (should detect differences):"
if python "$COMPARE_SCRIPT" --dir1 "$RUN1_DIR" --dir2 "$CORRUPTED_DIR" --atol 1e-5 --rtol 1e-3; then
    echo "ERROR: compare.py failed to detect corrupted tensor!"
    exit 1
else
    echo "OK: compare.py correctly detected corrupted tensor"
fi

# 7. Cross-framework comparison (InfiniTrain vs PyTorch)
echo ""
echo "=== 7. Cross-framework comparison (InfiniTrain vs PyTorch) ==="
PYTORCH_DIR="${PYTORCH_TENSORS_DIR:-../pytorch_tensors}"

# Use single-rank test output for cross-framework comparison
SINGLE_RANK_DIR="$OUTPUT_DIR/test1_simple/$(ls -t "$OUTPUT_DIR/test1_simple" | head -1)/rank_0"

if [ -d "$PYTORCH_DIR" ] && [ "$(ls -A "$PYTORCH_DIR" 2>/dev/null)" ]; then
    echo "PyTorch tensors directory: $PYTORCH_DIR"
    echo "InfiniTrain tensors directory: $SINGLE_RANK_DIR"
    echo "Comparing outputs:"
    python "$COMPARE_SCRIPT" --dir1 "$SINGLE_RANK_DIR" --dir2 "$PYTORCH_DIR" --atol 1e-5 --rtol 1e-3 || true
else
    echo "Skipping: PyTorch tensors directory not found or empty ($PYTORCH_DIR)"
    echo "To enable this test, run PyTorch with precision hook first:"
    echo "  python pytorch_llama3.py --precision_hook $PYTORCH_DIR ..."
fi

echo ""
echo "=== Verification Complete ==="
echo "Test output directory: $OUTPUT_DIR"
