#!/usr/bin/env python3
"""评估入口 — 计算已有模型的mAP等指标。

用法:
    python scripts/eval.py --task example_coco8
    python scripts/eval.py --task weld_defect --model-size m
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from utils.config_loader import load_config, get_task_data_path
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionCore 评估脚本")
    parser.add_argument("--task", type=str, required=True, help="任务名称")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["n", "s", "m", "l", "x"], help="模型大小 (覆盖配置文件)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.task)
    task_name = cfg["task"]["name"]

    data_path = get_task_data_path(cfg)
    model_size = args.model_size or cfg["model"]["size"]

    model_dir = Path(__file__).resolve().parent.parent / "models" / task_name
    weight_path = model_dir / "weights" / "best.pt"

    if not weight_path.exists():
        logger.error("模型权重不存在: %s (请先运行 train.py)", weight_path)
        sys.exit(1)

    logger.info("加载模型: %s", weight_path)
    model = YOLO(str(weight_path))

    logger.info("开始评估: data=%s", data_path)
    results = model.val(data=data_path, imgsz=cfg["train"]["imgsz"])

    logger.info("评估完成!")
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info("mAP50: %.4f", metrics.get("metrics/mAP50(B)", 0))
        logger.info("mAP50-95: %.4f", metrics.get("metrics/mAP50-95(B)", 0))


if __name__ == "__main__":
    main()
