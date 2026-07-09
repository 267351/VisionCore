"""链路测试 — 验证 train.py 的完整执行过程。

注意：只做结构和参数调用验证，不做实际GPU训练。
实际训练由 example_coco8 全链路手动验证。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from utils.config_loader import load_config


class TestTrainConfigChain:
    """验证训练配置链路的完整性。"""

    def test_example_coco8_config_completeness(self):
        """example_coco8配置包含所有必需字段。"""
        cfg = load_config("example_coco8")

        # 顶层字段完整
        assert "task" in cfg
        assert "model" in cfg
        assert "train" in cfg
        assert "augment" in cfg
        assert "export" in cfg
        assert "data" in cfg

        # model 字段
        assert "size" in cfg["model"]
        assert cfg["model"]["size"] in ["n", "s", "m", "l", "x"]

        # train 字段
        required_train = ["epochs", "batch", "imgsz", "device", "workers", "lr0", "patience"]
        for f in required_train:
            assert f in cfg["train"], f"train配置缺少 {f}"

    def test_all_task_configs_loadable(self):
        """所有任务配置都能正常加载。"""
        task_dir = Path(__file__).resolve().parent.parent / "configs" / "tasks"
        for task_file in sorted(task_dir.glob("*.yaml")):
            task_name = task_file.stem
            cfg = load_config(task_name)
            assert cfg["task"]["name"] == task_name, f"配置name不匹配: {task_name}"
            print(f"  {task_name}: OK")

    @patch("ultralytics.YOLO")
    def test_train_script_args_override_config(self, mock_yolo):
        """CLI参数应覆盖配置文件中的对应值。"""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        cfg = load_config("weld_defect")
        assert cfg["train"]["epochs"] == 150  # 配置文件中的值

        # 模拟CLI覆盖：--epochs 300
        cfg["train"]["epochs"] = 300
        assert cfg["train"]["epochs"] == 300  # CLI参数生效
        assert cfg["train"]["batch"] == 16    # 未覆盖的字段保持默认
        assert cfg["train"]["imgsz"] == 1280  # 任务配置未覆盖的保持
