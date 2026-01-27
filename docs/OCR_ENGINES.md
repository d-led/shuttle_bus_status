# OCR Engine Comparison for License Plate Recognition

This document describes the available OCR engines and best practices for license plate recognition.

## Available OCR Engines

### 1. EasyOCR (Default)
- **Status**: Default, always available
- **Pros**: 
  - Easy to use, good general-purpose OCR
  - Supports 80+ languages
  - Works out of the box
- **Cons**:
  - May not be optimal for license plates specifically
  - Can be slower than specialized engines
- **Installation**: Already included in dependencies
- **Best for**: Quick setup, general use

### 2. PaddleOCR
- **Status**: Optional (install separately)
- **Pros**:
  - Often better accuracy for license plates
  - Fast inference
  - Good Chinese/English support, decent German
  - Active development
- **Cons**:
  - Requires separate installation
  - Larger model size
- **Installation**: 
  ```bash
  pip install paddlepaddle paddleocr
  ```
- **Best for**: Better accuracy on license plates

### 3. Tesseract OCR
- **Status**: Optional (install separately)
- **Pros**:
  - Classic, well-established OCR engine
  - Good for structured text (like license plates)
  - Highly configurable
  - System-level installation (can be faster)
- **Cons**:
  - Requires system-level Tesseract installation
  - May need language packs
- **Installation**:
  ```bash
  # macOS
  brew install tesseract tesseract-lang
  
  # Linux (Raspberry Pi)
  sudo apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng
  
  # Python package
  pip install pytesseract
  ```
- **Best for**: Classic approach, system integration

## Configuration

Set the OCR engine in `config.toml`:

```toml
[plate_recognition]
ocr_engine = "easyocr"  # or "paddleocr", "tesseract", "dotsocr", "chandra"
# Or use an ensemble:
# ocr_engine = ["dotsocr", "chandra", "easyocr"]  # Try multiple engines, pick best result
```

## Best Practices

### Preprocessing
**⚠️ DISABLED BY DEFAULT**: Test results show preprocessing does not improve accuracy and significantly slows down processing (5-15 seconds vs 0.5-2 seconds). Keep `preprocess = false` unless you have specific evidence it helps your use case.

Preprocessing may help with:
- Very small images (< 100px width)
- Low contrast images
- Blurry images

But it can hurt:
- Already good quality images
- Images with complex backgrounds
- Some OCR engines work better on original images

```toml
preprocess = false  # Start with false, enable only if it helps
```

### Allowlist
Restrict to valid characters:
```toml
allowlist = true
allowlist_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
```

### Normalization
Normalize output for consistency:
```toml
normalize = true  # Uppercase, strip separators
```

### Confidence Threshold
Adjust based on your needs:
```toml
min_confidence = 0.3  # Lower = more detections (may include false positives)
```

## Performance Comparison

Based on typical license plate datasets:
- **PaddleOCR**: Often highest accuracy for plates
- **EasyOCR**: Good balance of accuracy and ease of use
- **Tesseract**: Good for structured text, requires tuning

## Recommendations

1. **Start with EasyOCR** (default) - works out of the box, but may have lower accuracy on license plates
2. **Try PaddleOCR** if accuracy is insufficient - **often significantly better for license plates**, recommended for production
3. **Use Tesseract** if you need system-level integration or have specific requirements

## Troubleshooting Low Accuracy

If you're seeing low OCR accuracy (e.g., < 10%):

1. **Try PaddleOCR**: Often 2-3x better accuracy than EasyOCR for license plates
   ```bash
   pip install paddlepaddle paddleocr
   # Then set ocr_engine = "paddleocr" in config.toml
   ```

2. **Check image quality**: Very small images (< 100px width) or low-quality images will have poor OCR results

3. **Preprocessing**: `preprocess = false` (default) - Test results show no accuracy improvement, and it slows processing significantly

4. **Adjust confidence threshold**: Lower `min_confidence` to see more results (but may include false positives)

5. **Check the diagnostic script**: Run `python scripts/diagnose_ocr.py` to see what OCR is actually reading

## Testing

Run the accuracy test to compare engines:
```bash
# Test with EasyOCR (default)
scripts/test_datasets.sh

# Change engine in config.toml and test again
# Compare results in reports/datasets/ocr_accuracy_report.html
```
