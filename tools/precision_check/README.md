# Precision Check Tools

Tools for precision checking and tensor comparison in InfiniTrain.

## Directory Structure

```
tools/precision_check/
├── compare.py       # Tensor comparison tool
├── pytorch_hook.py  # PyTorch tensor capture hook
├── run_gpt2.sh      # GPT-2 validation script
├── run_llama3.sh    # LLaMA3 validation script
└── README.md
```

---

## InfiniTrain Precision Checker

Runtime precision checking for detecting NaN/Inf and capturing tensor statistics.

### Features

- **NaN/Inf Detection**: Automatic detection during forward and backward passes
- **Multi-level Checking**: MODULE level (1) or FUNCTION level (2)
- **Output Formats**: Simple (statistics) or MD5 (hash for quick comparison)
- **NPY Export**: Save tensors as .npy files for offline analysis
- **Multi-rank Support**: Each rank outputs to separate `rank_N` directory

### Configuration

```bash
--precision_check "level=1,path=./output,format=simple,save_tensors=true"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | int | 0 | 0=OFF, 1=MODULE, 2=FUNCTION |
| `path` | string | `./precision_check` | Output directory |
| `format` | string | `simple` | `simple` or `md5` |
| `save_tensors` | bool | false | Save tensors as .npy files |
| `md5_tolerance` | double | 0.0 | MD5 quantization tolerance (e.g., 1e-3). 0 = no quantization |

> **MD5 Tolerance**: When `md5_tolerance` is set, tensor values are quantized before MD5 calculation.
> For example, with `md5_tolerance=1e-3`, values `4.0003` and `4.0004` both become `4.000`,
> producing the same MD5 hash. This is useful for comparing BF16/FP16 outputs where small precision differences are expected.

### Output Directory Structure

```
precision_check/
└── 20260122_143052/              # Timestamp (YYYYMMDD_HHMMSS)
    ├── precision_check_rank_0.log
    ├── rank_0/
    │   ├── transformer.h.0.ln_1_forward.npy
    │   ├── transformer.h.0.ln_1_backward.npy
    │   └── ...
    └── rank_1/
        └── ...
```

### Usage Examples

```bash
# Basic precision check
./build/gpt2 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1" \
    --num_iteration 1

# Save tensors as NPY files
./build/gpt2 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1,save_tensors=true" \
    --num_iteration 1

# MD5 format with tolerance (handles BF16/FP16 precision differences)
./build/gpt2 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1,format=md5,md5_tolerance=1e-3" \
    --num_iteration 1
```

---

## Tensor Comparison Tool

Compare tensor outputs between two directories (same-framework or cross-framework).

### Usage

```bash
python tools/precision_check/compare.py \
    --dir1 ./run1 --dir2 ./run2 \
    --atol 1e-5 --rtol 1e-3 \
    --verbose
```

**Arguments:**
- `--dir1`: First directory containing .npy files
- `--dir2`: Second directory containing .npy files
- `--atol`: Absolute tolerance (default: 1e-5)
- `--rtol`: Relative tolerance (default: 1e-3)
- `--verbose`: Show detailed output for passing tests

### Output Example

```
Directory 1: ./run1 (100 files)
Directory 2: ./run2 (100 files)
Tolerance: atol=1e-05, rtol=0.001

Comparing 100 common files...

FAIL: transformer.h.0.attn_forward.npy
  shape=(2, 1024, 768) dtype=float32
  max_abs=1.23e-03 mean_abs=4.56e-04
  max_rel=2.34e-03 mean_rel=5.67e-04

==================================================
Summary: 99 passed, 1 failed, 0 errors
Missing: 0 in dir1 only, 0 in dir2 only
```

---

## Cross-Framework Comparison

Compare InfiniTrain vs PyTorch outputs.

### Pre-Work：PyTorch Hook

Capture tensor outputs in PyTorch:

```python
from pytorch_hook import PrecisionHook

hook = PrecisionHook("./output_dir")
hook.register(model)
# ... run training ...
hook.remove()
```

### Step 1: Run PyTorch with precision hook

```bash
python pytorch_gpt2.py \
    --precision_hook ./pytorch_output \
    --num_iterations 1 \
    --batch_size 4 \
    --sequence_length 64
```

### Step 2: Run InfiniTrain with precision checking

```bash
./build/gpt2 \
    --precision_check "level=1,path=./infini_output,save_tensors=true" \
    --num_iteration 1 \
    --batch_size 4 \
    --sequence_length 64 \
    --device cuda
```

### Step 3: Compare outputs

```bash
python tools/precision_check/compare.py \
    --dir1 ./infini_output/<timestamp>/rank_0 \
    --dir2 ./pytorch_output \
    --atol 1e-5 --rtol 1e-3
```


---

## Validation Scripts

End-to-end validation scripts for GPT-2 and LLaMA3.

```bash
# GPT-2 validation
bash tools/precision_check/run_gpt2.sh

# LLaMA3 validation
bash tools/precision_check/run_llama3.sh
```

### Test Cases

1. **Single-rank test (Simple format)** - Basic precision check with statistics output
2. **Single-rank test (MD5 format)** - Precision check with MD5 hash output
3. **Multi-iteration overwrite test** - Verify files are overwritten across iterations
4. **Comparison test (same-framework)** - Compare two InfiniTrain runs
5. **Error detection test** - Verify compare.py can detect corrupted tensors
6. **Cross-framework comparison** - Compare InfiniTrain vs PyTorch (if PyTorch tensors available)
7. **Multi-rank test** - Distributed training precision check (if available)

### Environment Variables

- `PYTORCH_TENSORS_DIR`: PyTorch tensor output directory for cross-framework comparison (default: `../pytorch_tensors`)

```bash
# Custom PyTorch tensors directory
PYTORCH_TENSORS_DIR=/path/to/pytorch/output bash tools/precision_check/run_gpt2.sh
```

---

## Troubleshooting

### Ensure Consistent Configuration

Both frameworks must use identical:
- Batch size
- Sequence length
- Model configuration
- Random seed
- Data type (float32 recommended for comparison)

### Disable Non-Deterministic Operations

```python
# PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
```

---

## Related Files

- `infini_train/include/utils/precision_checker.h` - API definition
- `infini_train/include/utils/precision_check_config.h` - Configuration
- `infini_train/include/utils/precision_check_context.h` - Context tracking
