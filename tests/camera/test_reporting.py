from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from camera.plate_pipeline import BBox, ImagePlateDetections, PlateDetection
from camera.reporting import write_detection_report_html


def test_write_detection_report_html_writes_two_column_layout(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    result = ImagePlateDetections(
        captured_at=ts,
        image_size=(100, 50),
        image_path=tmp_path / "img.jpg",
        detections=[
            PlateDetection(
                captured_at=ts,
                bbox=BBox(1, 2, 3, 4),
                detection_confidence=0.9,
                text="M AB 1234",
                ocr_confidence=0.8,
                reliability=0.72,
                crop_jpeg_b64=None,
                metadata={"k": "v"},
            )
        ],
    )

    write_detection_report_html(out_path=out, results=[result], title="t")

    html = out.read_text(encoding="utf-8")
    assert "<title>t</title>" in html
    assert "grid-template-columns" in html  # 2-column layout
    assert "img.jpg" in html
    assert "M AB 1234" in html
