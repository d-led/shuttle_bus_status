#!/usr/bin/env python3
"""Parallel benchmark comparing current OCR vs CRNN (PaddleOCR) approach.

Tests both OCR engines in parallel on the same images to compare:
- Speed (processing time per image)
- Accuracy (success rate)
- FPS (frames per second)

This script runs both engines simultaneously using multiprocessing for true parallel execution.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.config import Settings


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single OCR engine."""

    engine: str
    status: str
    init_time: float
    num_images: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    total_time: float
    fps: float
    success_rate: float
    times: list[float]
    error: str | None = None


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
    import re

    name = Path(filename).stem
    match = re.search(r"([A-Z0-9]{3,10})", name.upper())
    if match:
        return match.group(1)
    return None


def benchmark_engine_worker(
    engine_name: str,
    image_paths: list[str],  # Use strings for multiprocessing compatibility
    settings_dict: dict[str, Any],
) -> BenchmarkResult:
    """Worker function to benchmark a single OCR engine (runs in separate process)."""
    try:
        # Re-import in worker process (multiprocessing requirement)
        import sys
        from pathlib import Path

        import cv2

        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from camera.plate_pipeline import create_plate_recognizer_from_config

        # Reconstruct settings from dict
        rec_settings = settings_dict["plate_recognition"]

        # Create recognizer
        init_start = time.perf_counter()
        ocr = create_plate_recognizer_from_config(
            ocr_engine=engine_name,
            languages=rec_settings["languages"],
            min_confidence=rec_settings["min_confidence"],
            preprocess=rec_settings["preprocess"],
            allowlist=rec_settings["allowlist"],
            allowlist_chars=rec_settings["allowlist_chars"],
            normalize=rec_settings["normalize"],
        )
        init_time = time.perf_counter() - init_start

        # Warm-up run
        if image_paths:
            test_img = cv2.imread(image_paths[0])
            if test_img is not None:
                ocr.recognize(test_img)

        # Benchmark runs
        times: list[float] = []
        results: list[tuple[str | None, float | None]] = []

        for img_path_str in image_paths:
            img = cv2.imread(img_path_str)
            if img is None:
                continue

            start = time.perf_counter()
            text, conf, _meta = ocr.recognize(img)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            results.append((text, conf))

        if not times:
            return BenchmarkResult(
                engine=engine_name,
                status="failed",
                init_time=init_time,
                num_images=0,
                avg_time_ms=0.0,
                min_time_ms=0.0,
                max_time_ms=0.0,
                total_time=0.0,
                fps=0.0,
                success_rate=0.0,
                times=[],
                error="No images processed",
            )

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        # Calculate success rate
        success_count = 0
        for img_path_str, (text, conf) in zip(image_paths, results):
            gt = extract_ground_truth(Path(img_path_str).name)
            if gt and text:
                gt_norm = "".join(c for c in gt.upper() if c.isalnum())
                text_norm = "".join(c for c in text.upper() if c.isalnum())
                if gt_norm == text_norm or gt_norm in text_norm or text_norm in gt_norm:
                    success_count += 1

        success_rate = success_count / len(image_paths) if image_paths else 0.0

        return BenchmarkResult(
            engine=engine_name,
            status="success",
            init_time=init_time,
            num_images=len(image_paths),
            avg_time_ms=avg_time * 1000,
            min_time_ms=min_time * 1000,
            max_time_ms=max_time * 1000,
            total_time=total_time,
            fps=fps,
            success_rate=success_rate,
            times=times,
        )

    except Exception as e:
        import traceback
        return BenchmarkResult(
            engine=engine_name,
            status="failed",
            init_time=0.0,
            num_images=0,
            avg_time_ms=0.0,
            min_time_ms=0.0,
            max_time_ms=0.0,
            total_time=0.0,
            fps=0.0,
            success_rate=0.0,
            times=[],
            error=f"{str(e)}\n{traceback.format_exc()}",
        )


def benchmark_parallel(
    current_engine: str,
    crnn_engine: str,
    images: list[Path],
    *,
    preprocess: bool = False,
    languages: list[str] | None = None,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Run both OCR engines in parallel."""
    settings = Settings.load_from_project_root()
    rec = settings.plate_recognition

    # Prepare settings dict for worker processes
    settings_dict = {
        "plate_recognition": {
            "languages": languages or rec.languages,
            "min_confidence": rec.min_confidence,
            "preprocess": preprocess,
            "allowlist": rec.allowlist,
            "allowlist_chars": rec.allowlist_chars,
            "normalize": rec.normalize,
        }
    }

    print(f"\n{'='*80}")
    print(f"PARALLEL BENCHMARK: {current_engine.upper()} vs {crnn_engine.upper()} (CRNN)")
    print(f"{'='*80}")
    print(f"Testing {len(images)} images in parallel...")
    print(f"Preprocessing: {preprocess}")

    # Run both engines in parallel
    # Convert Path objects to strings for multiprocessing compatibility
    image_paths_str = [str(img) for img in images]

    start_time = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_current = executor.submit(
            benchmark_engine_worker,
            current_engine,
            image_paths_str,
            settings_dict,
        )
        future_crnn = executor.submit(
            benchmark_engine_worker,
            crnn_engine,
            image_paths_str,
            settings_dict,
        )

        result_current = future_current.result()
        result_crnn = future_crnn.result()

    total_wall_time = time.perf_counter() - start_time

    print(f"\nParallel execution completed in {total_wall_time:.2f}s")
    print(f"(Sequential would take ~{result_current.total_time + result_crnn.total_time:.2f}s)")

    return result_current, result_crnn


def print_comparison(
    current_result: BenchmarkResult,
    crnn_result: BenchmarkResult,
) -> None:
    """Print detailed comparison of results."""
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")

    # Header
    print(f"{'Metric':<25} {current_result.engine.upper():<20} {crnn_result.engine.upper():<20} {'Winner':<15}")
    print("-" * 80)

    # Status
    print(
        f"{'Status':<25} "
        f"{current_result.status:<20} "
        f"{crnn_result.status:<20} "
        f"{'':<15}"
    )

    if current_result.status != "success" or crnn_result.status != "success":
        if current_result.error:
            print(f"  {current_result.engine} error: {current_result.error}")
        if crnn_result.error:
            print(f"  {crnn_result.engine} error: {crnn_result.error}")
        return

    # Initialization time
    init_winner = (
        current_result.engine
        if current_result.init_time < crnn_result.init_time
        else crnn_result.engine
    )
    print(
        f"{'Init time (s)':<25} "
        f"{current_result.init_time:<20.2f} "
        f"{crnn_result.init_time:<20.2f} "
        f"{init_winner:<15}"
    )

    # Average time
    speed_winner = (
        current_result.engine
        if current_result.avg_time_ms < crnn_result.avg_time_ms
        else crnn_result.engine
    )
    speed_ratio = (
        max(current_result.avg_time_ms, crnn_result.avg_time_ms)
        / min(current_result.avg_time_ms, crnn_result.avg_time_ms)
    )
    print(
        f"{'Avg time (ms)':<25} "
        f"{current_result.avg_time_ms:<20.1f} "
        f"{crnn_result.avg_time_ms:<20.1f} "
        f"{speed_winner} ({speed_ratio:.2f}x faster)"
    )

    # FPS
    fps_winner = (
        current_result.engine
        if current_result.fps > crnn_result.fps
        else crnn_result.engine
    )
    print(
        f"{'FPS':<25} "
        f"{current_result.fps:<20.2f} "
        f"{crnn_result.fps:<20.2f} "
        f"{fps_winner:<15}"
    )

    # Success rate
    accuracy_winner = (
        current_result.engine
        if current_result.success_rate > crnn_result.success_rate
        else crnn_result.engine
    )
    print(
        f"{'Success rate (%)':<25} "
        f"{current_result.success_rate*100:<20.1f} "
        f"{crnn_result.success_rate*100:<20.1f} "
        f"{accuracy_winner:<15}"
    )

    # Min/Max times
    print(f"\n{'Min time (ms)':<25} {current_result.min_time_ms:<20.1f} {crnn_result.min_time_ms:<20.1f}")
    print(f"{'Max time (ms)':<25} {current_result.max_time_ms:<20.1f} {crnn_result.max_time_ms:<20.1f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if speed_winner == crnn_result.engine:
        speedup = current_result.avg_time_ms / crnn_result.avg_time_ms
        print(f"\n✓ CRNN ({crnn_result.engine}) is {speedup:.2f}x FASTER than {current_result.engine}")
        print(f"  CRNN: {crnn_result.avg_time_ms:.1f}ms/image ({crnn_result.fps:.2f} FPS)")
        print(f"  Current: {current_result.avg_time_ms:.1f}ms/image ({current_result.fps:.2f} FPS)")
    else:
        slowdown = crnn_result.avg_time_ms / current_result.avg_time_ms
        print(f"\n✗ CRNN ({crnn_result.engine}) is {slowdown:.2f}x SLOWER than {current_result.engine}")
        print(f"  Current: {current_result.avg_time_ms:.1f}ms/image ({current_result.fps:.2f} FPS)")
        print(f"  CRNN: {crnn_result.avg_time_ms:.1f}ms/image ({crnn_result.fps:.2f} FPS)")

    if accuracy_winner == crnn_result.engine:
        acc_improvement = (
            (crnn_result.success_rate - current_result.success_rate) * 100
        )
        print(f"\n✓ CRNN ({crnn_result.engine}) has {acc_improvement:+.1f}% better accuracy")
    elif accuracy_winner == current_result.engine:
        acc_degradation = (
            (current_result.success_rate - crnn_result.success_rate) * 100
        )
        print(f"\n✗ CRNN ({crnn_result.engine}) has {acc_degradation:+.1f}% worse accuracy")
    else:
        print(f"\n= Both engines have similar accuracy ({current_result.success_rate*100:.1f}%)")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    if speed_winner == crnn_result.engine and accuracy_winner in [
        crnn_result.engine,
        "tie",
    ]:
        print(f"\n→ SWITCH to {crnn_result.engine} (CRNN)")
        print(f"  Reason: Faster ({speed_ratio:.2f}x) and similar or better accuracy")
    elif speed_winner == crnn_result.engine:
        print(f"\n→ CONSIDER {crnn_result.engine} (CRNN) for speed")
        print(f"  Faster ({speed_ratio:.2f}x) but accuracy is {abs((current_result.success_rate - crnn_result.success_rate)*100):.1f}% different")
    elif accuracy_winner == crnn_result.engine:
        print(f"\n→ CONSIDER {crnn_result.engine} (CRNN) for accuracy")
        print(f"  Better accuracy but {speed_ratio:.2f}x slower")
    else:
        print(f"\n→ KEEP {current_result.engine}")
        print("  Current engine is faster and has similar or better accuracy")


def main() -> int:
    """Run parallel OCR benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel benchmark: Current OCR vs CRNN (PaddleOCR)"
    )
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
        "--current-engine",
        type=str,
        default="easyocr",
        help="Current OCR engine to compare (default: easyocr)",
    )
    parser.add_argument(
        "--crnn-engine",
        type=str,
        default="paddleocr",
        help="CRNN-based OCR engine (default: paddleocr)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable preprocessing (slower but may improve accuracy)",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"Error: Dataset not found: {dataset_root}")
        return 1

    images = find_test_images(dataset_root, max_images=args.max_images)
    if not images:
        print(f"Error: No images found in {dataset_root}")
        return 1

    print(f"Found {len(images)} test images")
    print(f"Current engine: {args.current_engine}")
    print(f"CRNN engine: {args.crnn_engine}")

    # Run parallel benchmark
    current_result, crnn_result = benchmark_parallel(
        args.current_engine,
        args.crnn_engine,
        images,
        preprocess=args.preprocess,
    )

    # Print comparison
    print_comparison(current_result, crnn_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
