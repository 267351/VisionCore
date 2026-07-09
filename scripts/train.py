#!/usr/bin/env python3
"""训练入口 — 配置驱动的YOLOv8训练脚本。

用法:
    python scripts/train.py --task example_coco8
    python scripts/train.py --task weld_defect --epochs 200 --model-size m
    python scripts/train.py --task weld_defect --resume
"""

import argparse
import sys
from pathlib import Path

# 将 scripts/ 加入 sys.path 以支持 from utils import ...
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from utils.config_loader import load_config, get_task_data_path, is_builtin_dataset
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionCore 训练脚本")
    parser.add_argument("--task", type=str, required=True, help="任务名称 (对应 configs/tasks/{task}.yaml)")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["n", "s", "m", "l", "x"], help="YOLOv8模型大小 (覆盖配置文件)")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (覆盖配置文件)")
    parser.add_argument("--batch", type=int, default=None, help="批次大小 (覆盖配置文件)")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图片尺寸 (覆盖配置文件)")
    parser.add_argument("--device", type=int, default=None, help="GPU设备ID, -1为CPU (覆盖配置文件)")
    parser.add_argument("--resume", action="store_true", help="从中断处恢复训练")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载并合并配置
    cfg = load_config(args.task)
    task_name = cfg["task"]["name"]
    task_type = cfg["task"]["type"]

    # 2. CLI参数覆盖配置文件中的对应字段
    if args.model_size:
        cfg["model"]["size"] = args.model_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch is not None:
        cfg["train"]["batch"] = args.batch
    if args.imgsz is not None:
        cfg["train"]["imgsz"] = args.imgsz
    if args.device is not None:
        cfg["train"]["device"] = args.device

    model_size = cfg["model"]["size"]
    epochs = cfg["train"]["epochs"]
    imgsz = cfg["train"]["imgsz"]
    batch = cfg["train"]["batch"]
    device = cfg["train"]["device"]
    pretrained = cfg["model"]["pretrained"]

    # 3. 数据集路径
    data_path = get_task_data_path(cfg)
    builtin = is_builtin_dataset(cfg["data"]["path"])

    # 4. 输出目录
    model_dir = Path(__file__).resolve().parent.parent / "models" / task_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 5. 模型权重路径
    if args.resume:
        weight_path = model_dir / "train" / "weights" / "last.pt"
        if not weight_path.exists():
            logger.error("无法恢复训练: 未找到 last.pt (%s)", weight_path)
            sys.exit(1)
        logger.info("从 %s 恢复训练", weight_path)
        model = YOLO(str(weight_path))
    else:
        if pretrained:
            model_name = f"yolov8{model_size}{get_task_suffix(task_type)}.pt"
        else:
            model_name = f"yolov8{model_size}{get_task_suffix(task_type)}.yaml"
        logger.info("加载模型: %s", model_name)
        model = YOLO(model_name)

    # 6. 构建训练参数
    train_kwargs = {
        "data": data_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": cfg["train"]["workers"],
        "lr0": cfg["train"]["lr0"],
        "patience": cfg["train"]["patience"],
        "project": str(model_dir),
        "name": "train",
        "exist_ok": True,
        "resume": args.resume,
        # 数据增强参数
        **cfg.get("augment", {}),
    }

    logger.info("开始训练: task=%s, epochs=%d, imgsz=%d, batch=%d, device=%d",
                task_name, epochs, imgsz, batch, device)

    # 7. 训练
    results = model.train(**train_kwargs)

    # 8. 训练完成，输出摘要
    logger.info("训练完成!")
    logger.info("最佳模型: %s", model_dir / "train" / "weights" / "best.pt")
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info("mAP50: %.4f", metrics.get("metrics/mAP50(B)", 0))
        logger.info("mAP50-95: %.4f", metrics.get("metrics/mAP50-95(B)", 0))


def get_task_suffix(task_type: str) -> str:
    """根据任务类型返回 YOLO 模型后缀。"""
    suffixes = {
        "detect": "",
        "classify": "-cls",
        "segment": "-seg",
    }
    return suffixes.get(task_type, "")


if __name__ == "__main__":
    main()
