# Performance Optimization for Raspberry Pi

## Problem: Slow Detection Times

If you see warnings like:
```
Plate detection took 54.39s (interval=1.00s), skipping sleep
```

This indicates the detection pipeline is too slow for real-time use on Raspberry Pi 3.

## Common Causes

### 1. EasyOCR First-Time Initialization (30-60 seconds)

**First run only**: EasyOCR downloads and initializes models on first use. This is a one-time cost.

**Solution**: Wait for first initialization to complete. Subsequent runs will be much faster.

### 2. Multiple Preprocessing Strategies

**DISABLED BY DEFAULT**: Test results show preprocessing does not improve accuracy and significantly slows processing (5-15 seconds vs 0.5-2 seconds).

**Recommendation**: Keep `preprocess = false` in `config.toml`:
```toml
[plate_recognition]
preprocess = false  # DISABLED: No accuracy improvement, 5-10x slower
```

### 3. Large Image Resolution

1920x1080 images are large for detection, especially on CPU.

**Solution**: Reduce camera resolution in `config.toml`:
```toml
[camera]
width = 1280   # or 640 for even faster processing
height = 720   # or 480
```

### 4. CPU-Only Inference

YOLO running on CPU is much slower than GPU.

**Solution**: 
- Raspberry Pi doesn't have GPU acceleration for YOLO
- Use YOLOv8 nano model (already configured)
- Consider reducing image resolution further

## Recommended RPi 3 Configuration

```toml
[camera]
width = 1280
height = 720

[plate_detection]
model_size = "nano"  # Smallest, fastest model
confidence_threshold = 0.5
poll_interval_seconds = 2.0  # Process every 2 seconds instead of 1

[plate_recognition]
ocr_engine = "easyocr"
preprocess = false  # CRITICAL: Disable for RPi performance
allowlist = true
normalize = true
```

## Expected Performance

With recommended settings (preprocessing disabled):
- **First run**: 30-60s (EasyOCR initialization - one time only)
- **Subsequent runs**: 0.5-2 seconds per detection (without preprocessing)
- **With preprocessing enabled**: 5-15 seconds per detection (not recommended - no accuracy benefit)

## Monitoring Performance

The system logs warnings when detection takes > 5 seconds:
```
Plate detection took X.XXs (detection + OCR)
```

Check logs to identify bottlenecks.

## Additional Optimizations

1. **Skip frames**: Process every Nth frame instead of every frame
2. **Lower confidence threshold**: Reduce false positives, but may miss some plates
3. **Use Tesseract instead of EasyOCR**: Often faster on CPU (but less accurate)
4. **Reduce OCR languages**: Only use `["de"]` instead of `["de", "en"]`

## Testing Performance

Run a test detection to measure actual performance:
```bash
python3 -m camera.main --test-performance
```

Or check the logs after running the system for a few minutes.
