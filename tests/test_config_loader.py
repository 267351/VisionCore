"""配置加载器测试。"""

import tempfile
from pathlib import Path

import pytest
import yaml

from utils.config_loader import load_config, _deep_merge, _validate_task_config, is_builtin_dataset


def test_deep_merge_shallow_override():
    """测试浅合并：override 字段覆盖 base 同名字段。"""
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"x": 10}, "c": 3}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["x"] == 10   # 覆盖
    assert result["b"]["y"] == 2    # 保留
    assert result["c"] == 3         # 新增


def test_deep_merge_new_top_level():
    """测试浅合并：override 中 base 不存在的顶级字段直接加入。"""
    base = {"a": 1}
    override = {"b": 2}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 2}


def test_deep_merge_nested_preserved():
    """测试嵌套字典中未覆盖的字段被保留。"""
    base = {"train": {"epochs": 100, "batch": 16}}
    override = {"train": {"epochs": 200}}
    result = _deep_merge(base, override)
    assert result["train"]["epochs"] == 200
    assert result["train"]["batch"] == 16


def test_validate_task_config_missing_task():
    """测试缺少 task 字段时报错。"""
    with pytest.raises(KeyError, match="缺少 'task' 字段"):
        _validate_task_config({"data": {}}, "test")


def test_validate_task_config_missing_type():
    """测试缺少 task.type 字段时报错。"""
    with pytest.raises(KeyError, match="缺少 'task.type' 字段"):
        _validate_task_config({"task": {"name": "test"}, "data": {}}, "test")


def test_is_builtin_dataset_true():
    """测试内置数据集识别。"""
    assert is_builtin_dataset("coco8.yaml") is True
    assert is_builtin_dataset("coco128.yaml") is True


def test_is_builtin_dataset_false():
    """测试本地数据集识别。"""
    assert is_builtin_dataset("data/weld_defect/") is False


def test_load_config_example_coco8():
    """测试加载 example_coco8 任务配置。"""
    cfg = load_config("example_coco8")
    assert cfg["task"]["name"] == "example_coco8"
    assert cfg["task"]["type"] == "detect"
    # 默认值被保留
    assert cfg["model"]["size"] == "n"
    assert cfg["model"]["pretrained"] is True
    # 任务覆盖值
    assert cfg["train"]["epochs"] == 3


def test_load_config_weld_defect():
    """测试加载 weld_defect 任务配置 — 验证epochs覆盖。"""
    cfg = load_config("weld_defect")
    assert cfg["task"]["name"] == "weld_defect"
    assert cfg["train"]["epochs"] == 150     # 覆盖了默认的100
    assert cfg["train"]["imgsz"] == 1280     # 覆盖了默认的640
    assert cfg["train"]["batch"] == 16       # 沿用默认值


def test_load_config_not_found():
    """测试加载不存在的任务配置时抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="任务配置"):
        load_config("nonexistent_task")
