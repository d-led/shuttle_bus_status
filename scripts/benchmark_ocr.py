#!/usr/bin/env python3
"""Benchmark OCR engines for speed and accuracy on license plate images.

Tests all available OCR engines on a set of test images and reports:
- Average processing time per image
- Total time
- FPS (frames per second)
- Success rate (if ground truth available)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.config import Settings
from camera.plate_pipeline import create_plate_recognizer_from_config


def find_test_images(dataset_root: Path, max_images: int = 20) -> list[Path]:
    """Find test images from the dataset."""
    images = []
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        images.extend(dataset_root.rglob(f"*{ext}"))
        if len(images) >= max_images:
            break
    return images[:max_images]


def extract_ground_truth(filename: str) -> str | None:
    """Extract ground truth from filename (e.g., 'LIP317W.png' -> 'LIP317W')."""
    # Remove extension
    name = Path(filename).stem
    
    # Try to extract plate-like pattern (alphanumeric, 3-10 chars)
    import re
    match = re.search(r"([A-Z0-9]{3,10})", name.upper())
    if match:
        return match.group(1)
    return None


def benchmark_engine(
    engine_name: str,
    images: list[Path],
    *,
    preprocess: bool = False,
    languages: list[str] | None = None,
) -> dict[str, object]:
    """Benchmark a single OCR engine."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_name} (preprocess={preprocess})")
    print(f"{'='*60}")
    
    try:
        settings = Settings.load_from_project_root()
        rec = settings.plate_recognition
        
        # Create recognizer
        print(f"Initializing {engine_name}...")
        init_start = time.perf_counter()
        ocr = create_plate_recognizer_from_config(
            ocr_engine=engine_name,
            languages=languages or rec.languages,
            min_confidence=rec.min_confidence,
            preprocess=preprocess,
            allowlist=rec.allowlist,
            allowlist_chars=rec.allowlist_chars,
            normalize=rec.normalize,
        )
        init_time = time.perf_counter() - init_start
        print(f"Initialization took: {init_time:.2f}s")
        
        # Warm-up run (first run is often slower)
        if images:
            test_img = cv2.imread(str(images[0]))
            if test_img is not None:
                print("Warm-up run...")
                ocr.recognize(test_img)
        
        # Benchmark runs
        times: list[float] = []
        results: list[tuple[str | None, float | None]] = []
        
        print(f"\nProcessing {len(images)} images...")
        for i, img_path in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Time recognition
            start = time.perf_counter()
            text, conf, meta = ocr.recognize(img)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            results.append((text, conf))
            
            if i % 5 == 0:
                print(f"  Processed {i}/{len(images)} images...")
        
        if not times:
            return {
                "engine": engine_name,
                "preprocess": preprocess,
                "status": "failed",
                "error": "No images processed",
            }
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        # Calculate success rate if we can extract ground truth
        success_count = 0
        for img_path, (text, conf) in zip(images, results):
            gt = extract_ground_truth(img_path.name)
            if gt and text:
                # Simple match (normalized)
                gt_norm = "".join(c for c in gt.upper() if c.isalnum())
                text_norm = "".join(c for c in text.upper() if c.isalnum())
                if gt_norm == text_norm or gt_norm in text_norm or text_norm in gt_norm:
                    success_count += 1
        
        success_rate = success_count / len(images) if images else 0.0
        
        return {
            "engine": engine_name,
            "preprocess": preprocess,
            "status": "success",
            "init_time": init_time,
            "num_images": len(images),
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "total_time": total_time,
            "fps": fps,
            "success_rate": success_rate,
            "times": times,
        }
        
    except Exception as e:
        return {
            "engine": engine_name,
            "preprocess": preprocess,
            "status": "failed",
            "error": str(e),
        }


def main() -> int:
    """Run OCR benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark OCR engines")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/test_images/german_plates/kaggle/dataset_final/test",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to test",
    )
    parser.add_argument(
        "--engines",
        type=str,
        nargs="+",
        default=["easyocr", "paddleocr", "tesseract"],
        help="OCR engines to benchmark",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Test without preprocessing (faster)",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Test only with preprocessing",
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"Error: Dataset not found: {dataset_root}")
        return 1
    
    # Find test images
    images = find_test_images(dataset_root, max_images=args.max_images)
    if not images:
        print(f"Error: No images found in {dataset_root}")
        return 1
    
    print(f"Found {len(images)} test images")
    print(f"Testing engines: {args.engines}")
    
    # Test configurations
    test_configs = []
    if not args.preprocess_only:
        test_configs.append(False)  # Without preprocessing
    if not args.no_preprocess:
        test_configs.append(True)  # With preprocessing
    
    # Run benchmarks
    all_results: list[dict[str, object]] = []
    
    for engine in args.engines:
        for preprocess in test_configs:
            result = benchmark_engine(engine, images, preprocess=preprocess)
            all_results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Engine':<15} {'Preprocess':<12} {'Status':<10} {'Avg (ms)':<12} {'FPS':<10} {'Success %':<10}")
    print("-" * 80)
    
    for result in all_results:
        if result["status"] == "success":
            print(
                f"{result['engine']:<15} "
                f"{str(result['preprocess']):<12} "
                f"{result['status']:<10} "
                f"{result['avg_time_ms']:>10.1f} "
                f"{result['fps']:>9.2f} "
                f"{result['success_rate']*100:>9.1f}%"
            )
        else:
            print(
                f"{result['engine']:<15} "
                f"{str(result['preprocess']):<12} "
                f"{result['status']:<10} "
                f"ERROR: {result.get('error', 'Unknown')}"
            )
    
    # Find fastest
    successful = [r for r in all_results if r["status"] == "success"]
    if successful:
        fastest = min(successful, key=lambda x: x["avg_time_ms"])
        print(f"\n{'='*80}")
        print(f"FASTEST: {fastest['engine']} (preprocess={fastest['preprocess']})")
        print(f"  Average: {fastest['avg_time_ms']:.1f}ms per image")
        print(f"  FPS: {fastest['fps']:.2f}")
        print(f"  Success rate: {fastest['success_rate']*100:.1f}%")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    # Fastest without preprocessing
    fastest_no_prep = min(
        [r for r in successful if not r["preprocess"]],
        key=lambda x: x["avg_time_ms"],
        default=None,
    )
    if fastest_no_prep:
        print("\nFor SPEED (no preprocessing):")
        print(f"  Use: {fastest_no_prep['engine']}")
        print("  Config: preprocess = false")
        print(f"  Expected: {fastest_no_prep['avg_time_ms']:.1f}ms per image ({fastest_no_prep['fps']:.2f} FPS)")
    
    # Fastest with preprocessing
    fastest_with_prep = min(
        [r for r in successful if r["preprocess"]],
        key=lambda x: x["avg_time_ms"],
        default=None,
    )
    if fastest_with_prep:
        print("\nFor ACCURACY (with preprocessing):")
        print(f"  Use: {fastest_with_prep['engine']}")
        print("  Config: preprocess = true")
        print(f"  Expected: {fastest_with_prep['avg_time_ms']:.1f}ms per image ({fastest_with_prep['fps']:.2f} FPS)")
        print(f"  WARNING: {fastest_with_prep['avg_time_ms']/fastest_no_prep['avg_time_ms']:.1f}x slower than no preprocessing")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
