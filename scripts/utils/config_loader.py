"""配置加载器 — 合并 default.yaml 和 task.yaml。

配置加载流程：
1. 加载 configs/default.yaml 作为基础配置
2. 加载 configs/tasks/{task_name}.yaml 作为任务配置
3. 任务配置中的字段浅覆盖默认配置中的同名字段
4. 返回合并后的完整配置
"""

from pathlib import Path
from typing import Any

import yaml

from .logger import get_logger

logger = get_logger(__name__)

# 项目根目录 — configs/ 的父目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """浅合并两个字典。override中的值覆盖base中的同名字段。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 嵌套字典递归合并
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(task_name: str) -> dict[str, Any]:
    """
    加载并合并任务配置。

    Args:
        task_name: 任务名称，对应 configs/tasks/{task_name}.yaml

    Returns:
        合并后的完整配置字典

    Raises:
        FileNotFoundError: 默认配置或任务配置文件不存在
        KeyError: 任务配置缺少必填字段
    """
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"默认配置文件不存在: {default_path}")

    task_path = PROJECT_ROOT / "configs" / "tasks" / f"{task_name}.yaml"
    if not task_path.exists():
        raise FileNotFoundError(f"任务配置文件不存在: {task_path}")

    with open(default_path, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)
        if default_cfg is None:
            raise ValueError(f"默认配置文件为空: {default_path}")

    with open(task_path, "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)
        if task_cfg is None:
            raise ValueError(f"任务配置文件为空: {task_path}")

    _validate_task_config(task_cfg, task_name)

    merged = _deep_merge(default_cfg, task_cfg)
    logger.info("配置加载完成: %s (类型: %s)", task_name, task_cfg["task"]["type"])
    return merged


def _validate_task_config(cfg: dict, task_name: str) -> None:
    """验证任务配置的必填字段。"""
    if "task" not in cfg:
        raise KeyError(f"任务配置缺少 'task' 字段: {task_name}")

    task = cfg["task"]
    required_fields = ["name", "type"]
    for field in required_fields:
        if field not in task:
            raise KeyError(f"任务配置缺少 'task.{field}' 字段: {task_name}")

    if "data" not in cfg:
        raise KeyError(f"任务配置缺少 'data' 字段: {task_name}")


def get_task_data_path(cfg: dict) -> str:
    """
    获取任务数据集路径。

    对于本地路径（以 data/ 开头），返回完整路径。
    对于 Ultralytics 内置数据集名（如 coco8.yaml），原样返回。
    """
    data_path = cfg["data"]["path"]
    if data_path.startswith("data/"):
        return str(PROJECT_ROOT / data_path)
    return data_path


def is_builtin_dataset(data_path: str) -> bool:
    """判断是否为 Ultralytics 内置数据集（不以 data/ 开头）。"""
    return not data_path.startswith("data/")
