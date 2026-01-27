# OCR Engine Benchmarking and Performance Optimization

## Quick Start

Run the benchmark to find the fastest OCR engine for your system:

```bash
# Test all available engines (with and without preprocessing)
./scripts/benchmark_ocr.py

# Test only without preprocessing (faster)
./scripts/benchmark_ocr.py --no-preprocess

# Test specific engines
./scripts/benchmark_ocr.py --engines easyocr tesseract

# Test on different dataset
./scripts/benchmark_ocr.py --dataset data/test_images/german_plates/kaggle/dataset_final/test
```

## Performance Optimizations

### 1. Adaptive Preprocessing (NEW)

The preprocessing system now uses **adaptive strategy selection** to reduce processing time:

- **Normal images** (≥50px height, ≥150px width):
  - Tries 3-4 fast strategies first
  - **Early exit** if confidence ≥ 0.7 and text length ≥ 4
  - Falls back to 2 more strategies if needed
  - **Result**: ~3-4x faster than before (was 8 strategies always)

- **Small images** (30-50px height, 100-150px width):
  - Tries 5-6 strategies (skips slowest ones)
  - **Result**: ~1.5x faster than before

- **Very small images** (<30px height or <100px width):
  - Uses all 8 strategies (needs maximum help)
  - **Result**: Same as before, but only for challenging cases

### 2. Recommended Configurations

#### For Speed (Raspberry Pi 3)

```toml
[plate_recognition]
ocr_engine = "tesseract"  # Usually fastest on CPU
preprocess = false  # CRITICAL: Disable for speed
```

#### Recommended Configuration (All Cases)

```toml
[plate_recognition]
ocr_engine = "easyocr"  # or "tesseract" for speed
preprocess = false  # DISABLED: Test results show no accuracy improvement
```

**Note**: Preprocessing is disabled by default because test results show it doesn't improve accuracy but makes processing 5-10x slower.

## Expected Performance

Based on typical results:

| Engine | No Preprocessing | With Preprocessing (Adaptive) | Notes |
|--------|------------------|------------------------------|-------|
| **tesseract** | 50-200ms | 200-800ms | Fastest, but lower accuracy |
| **easyocr** | 200-500ms | 500-2000ms | Good balance |
| **paddleocr** | 300-600ms | 800-2500ms | Better accuracy |
| **dotsocr** | 400-800ms | 1000-3000ms | High accuracy, requires GPU |
| **chandra** | 500-1000ms | 1500-4000ms | Highest accuracy, requires GPU |

**With adaptive preprocessing optimization:**
- Normal images: 3-4x faster than before
- Small images: 1.5x faster than before
- Very small images: Same as before (all strategies)

## Benchmark Results Format

The benchmark script outputs:

```
Engine          Preprocess   Status     Avg (ms)    FPS        Success %
--------------------------------------------------------------------------------
easyocr         False        success    245.3       4.08       65.0%
easyocr         True         success    892.1       1.12       78.5%
tesseract       False        success    89.2        11.21      45.0%
tesseract       True         success    312.5       3.20       52.5%
```

## Troubleshooting Slow OCR

If OCR is still taking 20-50 seconds:

1. **Check preprocessing**: Set `preprocess = false` in config.toml
2. **Run benchmark**: `./scripts/benchmark_ocr.py --no-preprocess` to find fastest engine
3. **Check initialization**: First run initializes models (30-60s one-time cost)
4. **Reduce image size**: Lower camera resolution in config.toml
5. **Use faster engine**: Switch to tesseract or easyocr without preprocessing

## Performance Monitoring

The system now logs separate timing for YOLO and OCR:

```
INFO: Detection timing: YOLO=0.234s, OCR=1.456s, total=1.690s (candidates=2)
```

This helps identify if the bottleneck is detection or recognition.
