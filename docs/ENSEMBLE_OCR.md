# Ensemble OCR Configuration

## Overview

The license plate recognition system supports **ensemble OCR**, where multiple OCR engines can be configured to run on the same image and the best result is automatically selected.

## Configuration

### Single Engine (Default)

```toml
[plate_recognition]
ocr_engine = "easyocr"
```

### Ensemble (Multiple Engines)

```toml
[plate_recognition]
# Try multiple engines, pick the best result
ocr_engine = ["dotsocr", "chandra", "easyocr"]
```

## How It Works

1. **Parallel Execution**: All configured OCR engines process the same plate image
2. **Result Collection**: Each engine returns:
   - Recognized text
   - Confidence score
   - Metadata (raw text, preprocessing info, etc.)
3. **Intelligent Selection**: The ensemble uses a scoring system to pick the best result:
   - **Confidence Score (40%)**: Higher confidence = better
   - **Text Quality (30%)**: Validates plate format (length, alphanumeric, uppercase)
   - **Consensus (20%)**: If multiple engines agree, that's a strong signal
   - **Engine Reliability (10%)**: Known high-quality engines (dotsocr, chandra) get bonus
4. **Best Result**: The result with the highest composite score is selected
5. **Fallback**: If all engines fail, returns `None` with metadata containing all results

## Available OCR Engines

| Engine | Accuracy | Speed | GPU Required | Best For |
|--------|----------|-------|--------------|----------|
| **easyocr** | Medium (35%) | Fast | No | General use, CPU-friendly |
| **paddleocr** | Medium-High | Fast | No | License plates, alphanumeric |
| **tesseract** | Low-Medium | Fast | No | Classic OCR, well-established |
| **dotsocr** | High | Medium | Yes | High accuracy, multilingual |
| **chandra** | High | Medium | Yes | Complex layouts, handwriting |

## Recommended Configurations

### High Accuracy (GPU Available)

```toml
ocr_engine = ["dotsocr", "chandra", "paddleocr"]
```

**Why**: dots.ocr and Chandra are state-of-the-art models with excellent accuracy. PaddleOCR provides a good fallback.

### Balanced (CPU-Friendly)

```toml
ocr_engine = ["easyocr", "paddleocr", "tesseract"]
```

**Why**: All three work well on CPU, with different strengths. EasyOCR is fast, PaddleOCR is good for plates, Tesseract is reliable.

### Fast (Speed Priority)

```toml
ocr_engine = ["easyocr", "paddleocr"]
```

**Why**: Two fast engines that complement each other.

### Single Engine Fallback

```toml
ocr_engine = ["dotsocr", "easyocr"]
```

**Why**: Try dots.ocr first for accuracy, fall back to easyocr if it fails.

## Performance Considerations

- **Speed**: Ensemble mode runs engines sequentially, so total time = sum of all engine times
- **Memory**: Each engine loads its models, so memory usage increases with more engines
- **GPU**: If using GPU engines (dotsocr, chandra), ensure sufficient GPU memory

## Selection Algorithm

The ensemble uses a **composite scoring system** to select the best result:

### Scoring Factors

1. **Confidence Score (40% weight)**
   - Uses the engine's reported confidence
   - Defaults: dotsocr/chandra = 0.7, others = 0.5

2. **Text Quality (30% weight)**
   - Validates license plate format:
     - Length: 5-10 chars = +0.2, 3-12 = +0.1
     - Alphanumeric ratio: higher = better
     - Uppercase ratio: higher = better (German plates are uppercase)

3. **Consensus (20% weight)**
   - If multiple engines agree (exact or 75%+ match), score increases
   - Consensus is a strong signal of correctness

4. **Engine Reliability (10% weight)**
   - Known high-quality engines get bonus:
     - dotsocr: 0.95
     - chandra: 0.90
     - paddleocr: 0.75
     - easyocr: 0.70
     - tesseract: 0.65

### Example Scoring

```
Result 1: "ABC123" from dotsocr (conf=0.9)
  Score = 0.9*0.4 + 1.0*0.3 + 0.0*0.2 + 0.95*0.1 = 0.755

Result 2: "ABC123" from easyocr (conf=0.8)  
  Score = 0.8*0.4 + 1.0*0.3 + 0.5*0.2 + 0.70*0.1 = 0.730

Result 3: "abc123" from tesseract (conf=0.7)
  Score = 0.7*0.4 + 0.9*0.3 + 0.0*0.2 + 0.65*0.1 = 0.615

→ Result 1 (dotsocr) wins with highest score
```

## Debugging

The ensemble stores all results in metadata:

```python
{
    "ensemble_engine": "dotsocr",  # Which engine was selected
    "ensemble_score": 0.92,  # Composite score (0-1) for the selected result
    "ensemble_rankings": [
        {"engine": "dotsocr", "score": 0.92, "text": "ABC123"},
        {"engine": "easyocr", "score": 0.78, "text": "ABC123"},
        {"engine": "chandra", "score": 0.65, "text": "ABC12"}
    ],
    "ensemble_results": [
        {
            "engine": "dotsocr",
            "text": "ABC123",
            "confidence": 0.95,
            "meta": {...}
        },
        {
            "engine": "easyocr",
            "text": "ABC123",
            "confidence": 0.87,
            "meta": {...}
        }
    ]
}
```

This allows you to:
- See which engine performed best
- Compare results across engines
- Debug why certain engines failed

## Example Usage

### Config File

```toml
[plate_recognition]
# Ensemble: try dots.ocr first, then easyocr as fallback
ocr_engine = ["dotsocr", "easyocr"]
languages = ["de", "en"]
min_confidence = 0.5
preprocess = false
allowlist = true
normalize = true
```

### Python Code

```python
from camera.plate_pipeline import create_plate_recognizer_from_config
from camera.config import Settings

settings = Settings.load_from_project_root()
rec = settings.plate_recognition

# Automatically handles single engine or ensemble
ocr = create_plate_recognizer_from_config(
    ocr_engine=rec.ocr_engine,  # Can be string or list
    languages=rec.languages,
    min_confidence=rec.min_confidence,
    preprocess=rec.preprocess,
    allowlist=rec.allowlist,
    normalize=rec.normalize,
)

# Use OCR as normal - ensemble is transparent
text, confidence, meta = ocr.recognize(plate_image)
print(f"Result: {text} (confidence: {confidence})")
print(f"Selected engine: {meta.get('ensemble_engine')}")
```

## Installation

Install the engines you want to use:

```bash
# Basic engines (CPU-friendly)
pip install easyocr paddlepaddle paddleocr pytesseract

# Advanced engines (GPU recommended)
pip install dots-ocr chandra-ocr
```

## Troubleshooting

### "Failed to initialize OCR engine"

- Check that the engine is installed: `pip list | grep -E "easyocr|paddleocr|dots-ocr|chandra"`
- Check import errors in logs
- The ensemble will continue with other engines if one fails

### Low Accuracy with Ensemble

- Try different engine combinations
- Check which engine is being selected (see metadata)
- Consider preprocessing settings
- Verify image quality

### Slow Performance

- Reduce number of engines in ensemble
- Use CPU-friendly engines only
- Consider GPU acceleration for dots.ocr/chandra

## Best Practices

1. **Start Simple**: Use a single engine first, add ensemble if needed
2. **Test Combinations**: Different image types may benefit from different ensembles
3. **Monitor Performance**: Check which engines are actually being selected
4. **Balance Speed/Accuracy**: More engines = higher accuracy but slower
5. **GPU Availability**: Use GPU engines (dotsocr, chandra) only if GPU is available
