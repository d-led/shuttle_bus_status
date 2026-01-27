# YOLO Model Weights

This directory is for YOLOv8 model weight files (`.pt` files) used for license plate detection.

## Auto-Download

**You don't need to manually download models!** The Ultralytics YOLO library will automatically download the required model weights on first use if they're not found locally.

When you run the system for the first time:
1. The code looks for `models/yolov8n.pt` (or other sizes based on config)
2. If not found, Ultralytics YOLO automatically downloads it from `ultralytics.com`
3. The model is cached in `~/.ultralytics/weights/` for future use

## Model Sizes

The system supports different YOLOv8 model sizes (configured in `config.toml`):

- **nano** (`yolov8n.pt`) - Fastest, ~6MB, lower accuracy
- **small** (`yolov8s.pt`) - Balanced, ~22MB
- **medium** (`yolov8m.pt`) - Better accuracy, ~52MB
- **large** (`yolov8l.pt`) - High accuracy, ~87MB
- **xlarge** (`yolov8x.pt`) - Best accuracy, ~136MB

Default: `nano` (configured in `config.toml` under `[plate_detection]`)

## Manual Download (Optional)

If you prefer to download models manually or want to use a custom-trained model:

1. **Pre-trained models**: Download from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. **Custom models**: Place your trained `.pt` file here and update `config.toml` to point to it

## License

YOLOv8 models are licensed under **AGPL-3.0**. See:
- [Ultralytics License](https://www.ultralytics.com/license)
- [YOLOv8 Model Licensing](https://roboflow.com/model-licenses/yolov8)

For commercial use without AGPL restrictions, consider:
- Roboflow commercial license (included with paid plans)
- Direct licensing from Ultralytics

## Note

Model files (`.pt`) are excluded from git via `.gitignore` to keep the repository size manageable. They will be auto-downloaded when needed.
