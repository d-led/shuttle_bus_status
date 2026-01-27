"""Reporting helpers for visual inspection of detections."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from camera.plate_pipeline import ImagePlateDetections, PlateDetection  # noqa: TC001


def write_detection_report_html(
    *,
    out_path: Path,
    results: list[ImagePlateDetections],
    title: str = "Plate detection report",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _render_html(title=title, results=results, report_dir=out_path.parent)
    out_path.write_text(html, encoding="utf-8")


def write_detection_report_md(
    *,
    out_path: Path,
    results: list[ImagePlateDetections],
    title: str = "Plate detection report",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = _render_markdown(title=title, results=results, report_dir=out_path.parent)
    out_path.write_text(md, encoding="utf-8")


def _render_html(
    *, title: str, results: list[ImagePlateDetections], report_dir: Path
) -> str:
    rows = "\n".join(_render_html_row(r, report_dir=report_dir) for r in results)
    # Get current timestamp in local timezone
    import time

    now = datetime.now(tz=None)  # Use local timezone
    tz_name = (
        time.tzname[time.daylight] if time.daylight is not None else time.tzname[0]
    )
    report_timestamp = now.strftime("%Y-%m-%d %H:%M:%S") + f" {tz_name}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    body {{ font-family: var(--sans); margin: 24px; color: #111; }}
    h1 {{ margin: 0 0 16px 0; font-size: 20px; }}
    .timestamp {{ font-family: var(--mono); font-size: 12px; color: #6b7280; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: minmax(240px, 520px) 1fr; gap: 16px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fff; }}
    .img {{ border-radius: 8px; border: 1px solid #f3f4f6; object-fit: contain; display: block; }}
    .img-main {{ width: 100%; max-height: 420px; background: #f9fafb; }}
    .img-crop {{ width: 100%; max-width: 420px; max-height: 180px; background: #f9fafb; }}
    .meta {{ font-family: var(--mono); font-size: 12px; white-space: pre-wrap; }}
    .path {{ font-family: var(--mono); font-size: 12px; color: #374151; word-break: break-all; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f3f4f6; font-size: 12px; }}
    .row {{ margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h1>{escape(title)} <span class="badge">{len(results)} images</span></h1>
  <div class="timestamp">Report generated: {report_timestamp}</div>
  {rows}
</body>
</html>
"""


def _render_html_row(result: ImagePlateDetections, *, report_dir: Path) -> str:
    img_html = "<div class='card'><div class='path'>(no image path)</div></div>"
    if result.image_path is not None:
        rel = _safe_relpath(result.image_path, report_dir)
        w, h = result.image_size
        img_html = (
            "<div class='card'>"
            f"<div class='path'>{escape(str(rel))}</div>"
            f"<img class='img img-main' src='{escape(rel.as_posix())}' loading='lazy' width='{w}' height='{h}' />"
            "</div>"
        )

    details = _result_as_pretty_json(result)
    crops = _render_crop_previews(result)
    return (
        "<div class='row grid'>"
        f"{img_html}"
        "<div class='card'>"
        f"{crops}"
        f"<div class='meta'>{escape(details)}</div>"
        "</div>"
        "</div>"
    )


def _render_markdown(
    *, title: str, results: list[ImagePlateDetections], report_dir: Path
) -> str:
    lines: list[str] = [f"# {title}", "", f"- Images: **{len(results)}**", ""]
    for r in results:
        lines.append("<table>")
        lines.append("<tr>")
        lines.append("<td style='width: 55%; vertical-align: top;'>")
        if r.image_path is not None:
            rel = _safe_relpath(r.image_path, report_dir)
            lines.append(f"<div><code>{escape(rel.as_posix())}</code></div>")
            lines.append(
                f"<img src='{escape(rel.as_posix())}' style='max-width: 100%; height: auto;' />"
            )
        else:
            lines.append("<div><em>(no image path)</em></div>")
        lines.append("</td>")
        lines.append("<td style='vertical-align: top;'>")
        lines.append("<pre><code>")
        lines.append(escape(_result_as_pretty_json(r)))
        lines.append("</code></pre>")
        lines.append("</td>")
        lines.append("</tr>")
        lines.append("</table>")
        lines.append("")
    return "\n".join(lines)


def _result_as_pretty_json(result: ImagePlateDetections) -> str:
    payload = {
        "captured_at": _iso(result.captured_at),
        "image_size": list(result.image_size),
        "image_path": str(result.image_path) if result.image_path is not None else None,
        "plate_count": result.plate_count,
        "detections": [_detection_to_dict(d) for d in result.detections],
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def _render_crop_previews(result: ImagePlateDetections) -> str:
    items: list[str] = []
    for idx, det in enumerate(result.detections):
        if not det.crop_jpeg_b64:
            continue
        label = f"crop[{idx}]"
        items.append(
            "<div style='margin-bottom: 10px;'>"
            f"<div class='path'>{escape(label)} (bbox={escape(str(det.bbox))})</div>"
            f"<img class='img img-crop' src='data:image/jpeg;base64,{escape(det.crop_jpeg_b64)}' loading='lazy' width='320' />"
            "</div>"
        )
    return "".join(items)


def _detection_to_dict(d: PlateDetection) -> dict[str, Any]:
    raw = asdict(d) if is_dataclass(d) else dict(d)
    raw["captured_at"] = _iso(d.captured_at)
    raw["bbox"] = asdict(d.bbox)
    return raw


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _safe_relpath(path: Path, base: Path) -> Path:
    # Use os.path.relpath so that paths outside the report directory still become
    # usable relative paths (e.g. reports/... -> ../../data/...).
    try:
        rel = os.path.relpath(path.resolve(), start=base.resolve())
        return Path(rel)
    except Exception:
        return path
