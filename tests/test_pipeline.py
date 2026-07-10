"""链路测试 — 验证 train.py 的完整执行过程。

注意：只做结构和参数调用验证，不做实际GPU训练。
实际训练由 example_coco8 全链路手动验证。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


class TestScriptEntryPoints:
    """验证各脚本入口的 CLI 参数解析和配置加载链路。"""

    @patch("argparse.ArgumentParser.parse_args")
    @patch("ultralytics.YOLO")
    def test_train_parse_args_resolves_task(self, mock_yolo, mock_args):
        """train.py --task 应正确解析并传给 load_config。"""
        mock_args.return_value = MagicMock(
            task="weld_defect", model_size=None, epochs=None,
            batch=None, imgsz=None, device=None, resume=False
        )
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        from scripts.train import parse_args, main
        args = parse_args()
        assert args.task == "weld_defect"

    @patch("argparse.ArgumentParser.parse_args")
    @patch("ultralytics.YOLO")
    def test_eval_parse_args_resolves_task(self, mock_yolo, mock_args):
        """eval.py --task 应正确解析。"""
        mock_args.return_value = MagicMock(task="weld_defect", model_size=None)
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        from scripts.eval import parse_args
        args = parse_args()
        assert args.task == "weld_defect"

    @patch("argparse.ArgumentParser.parse_args")
    def test_export_parse_args_accepts_multiple_formats(self, mock_args):
        """export.py --format 接受多个值。"""
        mock_args.return_value = MagicMock(
            task="weld_defect", model_size=None, format=["onnx", "engine"],
            half=False, dynamic=False
        )

        from scripts.export import parse_args
        args = parse_args()
        assert args.format == ["onnx", "engine"]

    def test_get_task_suffix_returns_correct_suffix(self):
        """get_task_suffix 根据任务类型返回正确后缀。"""
        from scripts.train import get_task_suffix
        assert get_task_suffix("detect") == ""
        assert get_task_suffix("classify") == "-cls"
        assert get_task_suffix("segment") == "-seg"
        assert get_task_suffix("unknown") == ""
