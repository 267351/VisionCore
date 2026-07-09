#!/usr/bin/env python3
"""导出入口 — 将训练好的模型导出为 ONNX/TensorRT 等格式。

用法:
    python scripts/export.py --task example_coco8
    python scripts/export.py --task weld_defect --format onnx engine --half
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionCore 模型导出脚本")
    parser.add_argument("--task", type=str, required=True, help="任务名称")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["n", "s", "m", "l", "x"], help="模型大小")
    parser.add_argument("--format", type=str, nargs="+", default=None,
                        help="导出格式 (onnx, engine, tflite), 默认读取配置文件")
    parser.add_argument("--half", action="store_true", help="FP16量化")
    parser.add_argument("--dynamic", action="store_true", help="动态batch尺寸")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.task)
    task_name = cfg["task"]["name"]

    model_size = args.model_size or cfg["model"]["size"]
    model_dir = Path(__file__).resolve().parent.parent / "models" / task_name
    weight_path = model_dir / "train" / "weights" / "best.pt"

    if not weight_path.exists():
        logger.error("模型权重不存在: %s (请先运行 train.py)", weight_path)
        sys.exit(1)

    logger.info("加载模型: %s", weight_path)
    model = YOLO(str(weight_path))

    formats = args.format or cfg["export"]["formats"]
    use_half = args.half or cfg["export"]["half"]
    use_dynamic = args.dynamic or cfg["export"]["dynamic"]

    export_dir = model_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        logger.info("导出中: format=%s, half=%s, dynamic=%s", fmt, use_half, use_dynamic)
        export_path = model.export(
            format=fmt,
            imgsz=cfg["export"]["imgsz"],
            half=use_half,
            dynamic=use_dynamic,
        )
        logger.info("导出成功: %s", export_path)


if __name__ == "__main__":
    main()
