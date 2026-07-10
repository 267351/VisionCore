#!/usr/bin/env python3
"""项目脚手架 — 一键创建新任务的数据目录和配置文件。

用法:
    python scripts/init_project.py --task rod_type --type classify --classes "E6013,E7018,309L,ER5356,ER70S6"
    python scripts/init_project.py --task weld_defect --type detect --classes "气孔,裂纹,咬边" --epochs 200 --imgsz 1280
    python scripts/init_project.py --task my_task --classes "class_a,class_b" --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_TEMPLATE = """task:
  name: "{task}"
  type: "{task_type}"
  description: "{desc}"

data:
  path: "data/{task}/"
  train: "images/train"
  val: "images/val"
  nc: {nc}
  names:
{names_block}

train:
  epochs: {epochs}
  imgsz: {imgsz}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionCore 新任务脚手架")
    parser.add_argument("--task", type=str, required=True, help="任务名称 (snake_case)")
    parser.add_argument("--type", type=str, default="detect",
                        choices=["detect", "classify", "segment"],
                        help="任务类型: detect(目标检测) / classify(分类) / segment(分割)")
    parser.add_argument("--classes", type=str, required=True,
                        help="类别列表, 逗号分隔, 如: 'E6013,E7018,309L'")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认100)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (默认640)")
    parser.add_argument("--model-size", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"], help="YOLO模型大小 (默认n)")
    parser.add_argument("--dry-run", action="store_true", help="仅预览, 不实际创建文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_name = args.task.strip()
    task_type = args.type.strip()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    if not classes:
        logger.error("--classes 不能为空")
        sys.exit(1)

    if " " in task_name or not task_name.islower():
        logger.warning("任务名建议使用 snake_case (如: rod_type, weld_defect)")

    nc = len(classes)
    desc = f"{task_name} — {task_type} 任务, {nc} 个类别"

    # 数据目录
    data_dirs = [
        PROJECT_ROOT / "data" / task_name / "images" / "train",
        PROJECT_ROOT / "data" / task_name / "images" / "val",
        PROJECT_ROOT / "data" / task_name / "labels" / "train",
        PROJECT_ROOT / "data" / task_name / "labels" / "val",
    ]

    # 配置文件路径
    config_path = PROJECT_ROOT / "configs" / "tasks" / f"{task_name}.yaml"

    # 生成配置内容
    names_block = "\n".join(f"    - {name}" for name in classes)
    config_content = CONFIG_TEMPLATE.format(
        task=task_name,
        task_type=task_type,
        desc=desc,
        nc=nc,
        names_block=names_block,
        epochs=args.epochs,
        imgsz=args.imgsz,
    )

    if args.dry_run:
        logger.info("=== 预览模式 (--dry-run) ===")
        logger.info("将创建以下目录:")
        for d in data_dirs:
            logger.info("  %s", d)
        logger.info("将创建配置文件: %s", config_path)
        logger.info("配置内容:\n%s", config_content)
        return

    # 创建目录
    for d in data_dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("创建目录: %s", d)

    # 写配置文件
    if config_path.exists():
        logger.error("配置文件已存在: %s (请先删除或用 --task 换名)", config_path)
        sys.exit(1)

    config_path.write_text(config_content, encoding="utf-8")
    logger.info("创建配置: %s", config_path)

    # 输出摘要
    logger.info("=" * 50)
    logger.info("任务 '%s' 创建完成!", task_name)
    logger.info("  类型: %s", task_type)
    logger.info("  类别数: %d (%s)", nc, ", ".join(classes))
    logger.info("  数据目录: data/%s/", task_name)
    logger.info("  配置文件: configs/tasks/%s.yaml", task_name)
    logger.info("=" * 50)
    logger.info("下一步:")
    logger.info("  1. 将训练图片放入: data/%s/images/train/", task_name)
    logger.info("  2. 将验证图片放入: data/%s/images/val/", task_name)
    logger.info("  3. 用 LabelImg 标注 (YOLO 格式), 放入 data/%s/labels/", task_name)
    logger.info("  4. 开始训练: python scripts/train.py --task %s", task_name)


if __name__ == "__main__":
    main()
