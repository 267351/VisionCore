#!/usr/bin/env python3
"""推理入口 — 用训练好的模型对新图片进行检测/分类。

用法:
    python scripts/infer.py --task example_coco8 --source photo.jpg
    python scripts/infer.py --task weld_defect --source ./test_images/ --conf 0.5
    python scripts/infer.py --task weld_defect --source 0  # 摄像头
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
    parser = argparse.ArgumentParser(description="VisionCore 推理脚本")
    parser.add_argument("--task", type=str, required=True, help="任务名称")
    parser.add_argument("--source", type=str, required=True, help="输入源: 图片路径/目录/摄像头编号(0)")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["n", "s", "m", "l", "x"], help="模型大小")
    parser.add_argument("--output", type=str, default=None, help="结果输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值, 默认0.25")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.task)
    task_name = cfg["task"]["name"]

    model_size = args.model_size or cfg["model"]["size"]
    model_dir = Path(__file__).resolve().parent.parent / "models" / task_name
    weight_path = model_dir / "weights" / "best.pt"

    if not weight_path.exists():
        logger.error("模型权重不存在: %s (请先运行 train.py)", weight_path)
        sys.exit(1)

    logger.info("加载模型: %s", weight_path)
    model = YOLO(str(weight_path))

    # 处理 source: 如果是数字字符串，转为 int（摄像头编号）
    source = args.source
    if source.isdigit():
        source = int(source)

    output_dir = args.output or str(model_dir / "inference")
    logger.info("推理中: source=%s, conf=%.2f, output=%s", args.source, args.conf, output_dir)

    results = model.predict(
        source=source,
        conf=args.conf,
        save=True,
        project=output_dir,
        name="predict",
        exist_ok=True,
    )

    logger.info("推理完成! 结果保存至: %s", output_dir)


if __name__ == "__main__":
    main()
