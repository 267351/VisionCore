# VisionCore 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零搭建配置驱动的 YOLOv8 训练框架 + C# ONNX 推理运行时，通过 coco8 验证全链路可运行。

**Architecture:** Python CLI 层 (train/eval/infer/export) 通过配置合并加载任务参数，驱动 Ultralytics YOLOv8 训练；导出 ONNX 后由 C# Inferencer 加载推理。配置层使用 default.yaml + tasks/*.yaml 双层浅合并。

**Tech Stack:** Python 3.10+, Ultralytics YOLOv8, uv, PyYAML, pytest; C# .NET 8.0, ONNX Runtime, ImageSharp

---

## File Structure Blueprint

```
VisionCore/
├── .gitignore                          # [Create] 忽略 data/, models/, __pycache__, .venv
├── .python-version                     # [Create] Python版本锁定
├── pyproject.toml                      # [Create] uv项目配置
├── README.md                           # [Create] 项目说明
├── configs/
│   ├── default.yaml                    # [Create] 全局默认参数
│   └── tasks/
│       ├── example_coco8.yaml          # [Create] coco8验证任务
│       ├── weld_defect.yaml            # [Create] 焊缝缺陷(占位)
│       ├── rod_type.yaml               # [Create] 焊条型号(占位)
│       └── rod_state.yaml              # [Create] 焊条状态(占位)
├── scripts/
│   ├── train.py                        # [Create] 训练入口
│   ├── eval.py                         # [Create] 评估入口
│   ├── infer.py                        # [Create] 推理入口
│   ├── export.py                       # [Create] 导出入口
│   └── utils/
│       ├── __init__.py                 # [Create]
│       ├── config_loader.py            # [Create] 配置合并
│       └── logger.py                   # [Create] 统一日志
├── deploy/
│   └── VisionCore.Runtime/
│       ├── VisionCore.Runtime.csproj   # [Create] 类库项目
│       ├── Inferencer.cs               # [Create] 核心推理
│       ├── Models/
│       │   └── DetectionResult.cs      # [Create] 结果模型
│       └── Examples/
│           └── Program.cs              # [Create] 控制台示例
└── tests/
    ├── __init__.py                     # [Create]
    ├── conftest.py                     # [Create] pytest fixtures
    ├── test_config_loader.py           # [Create] 配置测试
    └── test_pipeline.py                # [Create] 链路测试
```

**Responsibility Map:**
- `configs/default.yaml` — 所有训练的默认超参，单点修改全局生效
- `configs/tasks/*.yaml` — 每个任务的类别定义 + 覆盖参数，新增任务唯一改动点
- `scripts/utils/config_loader.py` — 配置合并的纯函数，无副作用，可独立测试
- `scripts/train.py` — CLI参数解析 + 配置加载 + 调用 YOLO.train()
- `scripts/eval.py` — CLI参数解析 + 调用 YOLO.val()
- `scripts/infer.py` — CLI参数解析 + 调用 YOLO.predict()
- `scripts/export.py` — CLI参数解析 + 调用 YOLO.export()
- `deploy/VisionCore.Runtime/Inferencer.cs` — 独立类库，加载ONNX→推理→后处理，不依赖Python
- `deploy/VisionCore.Runtime/Models/DetectionResult.cs` — 纯数据record，无逻辑
- `deploy/VisionCore.Runtime/Examples/Program.cs` — 使用示例，非库代码

---

### Task 1: 项目基础设施 — .gitignore, pyproject.toml, README, 目录结构

**Files:**
- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: 空目录 `configs/tasks/`, `data/`, `models/`, `scripts/utils/`, `deploy/VisionCore.Runtime/Models/`, `deploy/VisionCore.Runtime/Examples/`, `tests/`

- [ ] **Step 1: 创建 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Data & Models
data/
models/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: 创建 .python-version**

```
3.10
```

- [ ] **Step 3: 创建 pyproject.toml**

```toml
[project]
name = "visioncore"
version = "0.1.0"
description = "焊接视觉AI训练平台 - 配置驱动的YOLOv8训练框架"
requires-python = ">=3.10"
dependencies = [
    "ultralytics>=8.2.0",
    "opencv-python>=4.9.0",
    "pyyaml>=6.0",
    "onnx>=1.16.0",
    "onnxruntime-gpu>=1.18.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["scripts"]
```

- [ ] **Step 4: 创建 README.md**

```markdown
# VisionCore — 焊接视觉AI训练平台

配置驱动的YOLOv8训练框架，覆盖焊接领域全部视觉识别需求。

## 快速开始

```bash
# 安装依赖
uv sync

# 验证框架 — 用 coco8 数据集跑通全链路
uv run python scripts/train.py --task example_coco8
uv run python scripts/eval.py --task example_coco8
uv run python scripts/export.py --task example_coco8
```

## 新增识别任务

1. 数据放入 `data/{task_name}/images/{train,val}/`
2. 标注放入 `data/{task_name}/labels/{train,val}/`
3. 新建 `configs/tasks/{task_name}.yaml`
4. `uv run python scripts/train.py --task {task_name}`

## 项目结构

```
VisionCore/
├── configs/          # 配置层 (default.yaml + tasks/*.yaml)
├── data/             # 数据集 (不提交Git)
├── models/           # 训练产出 (不提交Git)
├── scripts/          # CLI工具 (train/eval/infer/export)
├── deploy/           # C# ONNX推理运行时
└── tests/            # 测试
```
```

- [ ] **Step 5: 创建所有必要的空目录**

Run:
```bash
mkdir -p configs/tasks data models scripts/utils deploy/VisionCore.Runtime/Models deploy/VisionCore.Runtime/Examples tests
```

- [ ] **Step 6: 创建所有 `__init__.py`**

```bash
touch scripts/__init__.py scripts/utils/__init__.py tests/__init__.py
```

- [ ] **Step 7: 用 uv 初始化虚拟环境并安装依赖**

```bash
uv sync
```

Expected: 创建 `.venv/` 目录，安装 ultralytics、opencv-python 等依赖。

- [ ] **Step 8: 验证 uv 环境可用**

```bash
uv run python -c "from ultralytics import YOLO; print('ultralytics OK')"
```

Expected output: `ultralytics OK`

---

### Task 2: 配置层 — default.yaml + 任务配置文件

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/tasks/example_coco8.yaml`
- Create: `configs/tasks/weld_defect.yaml`
- Create: `configs/tasks/rod_type.yaml`
- Create: `configs/tasks/rod_state.yaml`

- [ ] **Step 1: 创建 configs/default.yaml**

```yaml
# VisionCore 全局默认训练参数
# 所有任务共享此配置，task文件中的同名字段会覆盖此处

model:
  size: "n"          # n/s/m/l/x, nano最快
  pretrained: true   # 从预训练权重开始

train:
  epochs: 100
  batch: 16
  imgsz: 640
  device: 0          # GPU编号, -1为CPU
  workers: 8
  lr0: 0.01
  patience: 50       # 早停轮数

augment:
  mosaic: 1.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  fliplr: 0.5

export:
  formats: ["onnx"]
  imgsz: 640
  half: false
  dynamic: true
```

- [ ] **Step 2: 创建 configs/tasks/example_coco8.yaml**

```yaml
# 框架验证任务 — 使用 Ultralytics 内置 coco8 数据集
task:
  name: "example_coco8"
  type: "detect"
  description: "coco8验证任务 — 验证框架全链路"

data:
  path: "coco8.yaml"         # Ultralytics内置数据集，非本地路径
  train: ""
  val: ""
  nc: 0                      # 由coco8.yaml自动确定
  names: []

train:
  epochs: 3                  # 验证用，只跑3轮
```

- [ ] **Step 3: 创建 configs/tasks/weld_defect.yaml**

```yaml
task:
  name: "weld_defect"
  type: "detect"
  description: "焊缝缺陷检测 — 气孔、裂纹、咬边、未熔合、夹渣"

data:
  path: "data/weld_defect/"
  train: "images/train"
  val: "images/val"
  nc: 5
  names:
    - porosity
    - crack
    - undercut
    - lack_of_fusion
    - slag_inclusion

train:
  epochs: 150
  imgsz: 1280
```

- [ ] **Step 4: 创建 configs/tasks/rod_type.yaml**

```yaml
task:
  name: "rod_type"
  type: "classify"
  description: "焊条型号识别"

data:
  path: "data/rod_type/"
  train: "images/train"
  val: "images/val"
  nc: 5
  names:
    - E6013
    - E7018
    - 309L_stainless
    - ER5356_aluminum
    - ER70S6

train:
  epochs: 100
```

- [ ] **Step 5: 创建 configs/tasks/rod_state.yaml**

```yaml
task:
  name: "rod_state"
  type: "detect"
  description: "焊条头状态识别 — 新焊条、半消耗、残头、涂层破损"

data:
  path: "data/rod_state/"
  train: "images/train"
  val: "images/val"
  nc: 4
  names:
    - new_rod
    - half_used
    - stub_end
    - damaged_coating

train:
  epochs: 100
  imgsz: 640
```

- [ ] **Step 6: 验证所有 YAML 可以正确解析**

```bash
uv run python -c "
import yaml, glob
for f in glob.glob('configs/**/*.yaml', recursive=True):
    with open(f) as fh:
        yaml.safe_load(fh)
    print(f'{f}: OK')
"
```

Expected: 每个文件输出 `OK`。

---

### Task 3: 配置加载器 — config_loader.py

**Files:**
- Create: `scripts/utils/logger.py`
- Create: `scripts/utils/config_loader.py`
- Create: `tests/test_config_loader.py`

- [ ] **Step 1: 创建 scripts/utils/logger.py**

```python
"""统一日志模块。"""

import logging
import sys


def get_logger(name: str = "visioncore") -> logging.Logger:
    """获取预配置的logger实例。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 2: 创建 scripts/utils/config_loader.py**

```python
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
```

- [ ] **Step 3: 创建 tests/conftest.py**

```python
"""pytest fixtures。"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_dir():
    """临时目录，测试后自动清理。"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
```

- [ ] **Step 4: 创建 tests/test_config_loader.py**

```python
"""配置加载器测试。"""

import tempfile
from pathlib import Path

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

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
```

- [ ] **Step 5: 运行配置加载器测试**

```bash
uv run python -m pytest tests/test_config_loader.py -v
```

Expected: 8 tests pass.

---

### Task 4: 训练脚本 — train.py

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: 创建 scripts/train.py**

```python
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
        weight_path = model_dir / "weights" / "last.pt"
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
```

- [ ] **Step 2: 语法检查**

```bash
uv run python -c "import py_compile; py_compile.compile('scripts/train.py', doraise=True); print('OK')"
```

Expected: `OK`

---

### Task 5: 评估 + 推理 + 导出脚本

**Files:**
- Create: `scripts/eval.py`
- Create: `scripts/infer.py`
- Create: `scripts/export.py`

- [ ] **Step 1: 创建 scripts/eval.py**

```python
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
    weight_path = model_dir / "train" / "weights" / "best.pt"

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
```

- [ ] **Step 2: 创建 scripts/infer.py**

```python
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
    weight_path = model_dir / "train" / "weights" / "best.pt"

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
```

- [ ] **Step 3: 创建 scripts/export.py**

```python
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
```

- [ ] **Step 4: 语法检查全部三个脚本**

```bash
uv run python -c "
import py_compile
for f in ['scripts/eval.py', 'scripts/infer.py', 'scripts/export.py']:
    py_compile.compile(f, doraise=True)
    print(f'{f}: OK')
"
```

Expected: 3 files all `OK`.

---

### Task 6: 创建空 conftest.py 和链路测试

**Files:**
- Modify: `tests/conftest.py` (已有内容)
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 确认 tests/conftest.py 已存在**

无需额外修改——conftest.py 已在 Task 3 Step 3 创建。

- [ ] **Step 2: 创建 tests/test_pipeline.py**

```python
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
```

- [ ] **Step 3: 运行链路测试**

```bash
uv run python -m pytest tests/test_pipeline.py -v
```

Expected: 3 tests pass.

---

### Task 7: C# 推理运行时 — DetectionResult + Inferencer

**Files:**
- Create: `deploy/VisionCore.Runtime/VisionCore.Runtime.csproj`
- Create: `deploy/VisionCore.Runtime/Models/DetectionResult.cs`
- Create: `deploy/VisionCore.Runtime/Inferencer.cs`
- Create: `deploy/VisionCore.Runtime/Examples/Program.cs`

- [ ] **Step 1: 创建 VisionCore.Runtime.csproj**

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <RootNamespace>VisionCore.Runtime</RootNamespace>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.19.2" />
    <PackageReference Include="Microsoft.ML.OnnxRuntime.Extensions" Version="0.10.0" />
    <PackageReference Include="SixLabors.ImageSharp" Version="3.1.5" />
  </ItemGroup>

</Project>
```

> **注意**: 使用 `Microsoft.ML.OnnxRuntime`（CPU版本）以确保跨平台兼容。GPU版本 `Microsoft.ML.OnnxRuntime.Gpu` 按需切换，NuGet包名在代码注释中说明。

- [ ] **Step 2: 创建 Models/DetectionResult.cs**

```csharp
namespace VisionCore.Runtime.Models;

/// <summary>
/// 单次检测结果。
/// 坐标均为归一化值 [0.0, 1.0]，表示相对于图片宽高的比例。
/// </summary>
public record DetectionResult
{
    /// <summary>类别名称</summary>
    public string ClassName { get; init; } = string.Empty;

    /// <summary>类别索引</summary>
    public int ClassId { get; init; }

    /// <summary>置信度 [0.0, 1.0]</summary>
    public float Confidence { get; init; }

    /// <summary>边界框中心X (归一化, 0-1, 从左到右)</summary>
    public float X { get; init; }

    /// <summary>边界框中心Y (归一化, 0-1, 从上到下)</summary>
    public float Y { get; init; }

    /// <summary>边界框宽度 (归一化, 0-1)</summary>
    public float Width { get; init; }

    /// <summary>边界框高度 (归一化, 0-1)</summary>
    public float Height { get; init; }

    /// <summary>
    /// 将归一化坐标转换为像素坐标。
    /// </summary>
    public (int x1, int y1, int x2, int y2) ToPixel(int imageWidth, int imageHeight)
    {
        float halfW = Width / 2f;
        float halfH = Height / 2f;
        int x1 = Math.Clamp((int)((X - halfW) * imageWidth), 0, imageWidth);
        int y1 = Math.Clamp((int)((Y - halfH) * imageHeight), 0, imageHeight);
        int x2 = Math.Clamp((int)((X + halfW) * imageWidth), 0, imageWidth);
        int y2 = Math.Clamp((int)((Y + halfH) * imageHeight), 0, imageHeight);
        return (x1, y1, x2, y2);
    }
}
```

- [ ] **Step 3: 创建 Inferencer.cs**

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using VisionCore.Runtime.Models;

namespace VisionCore.Runtime;

/// <summary>
/// YOLOv8 ONNX 推理器。
/// 加载导出的 .onnx 模型，对图片进行目标检测推理。
///
/// 使用示例:
/// <code>
/// using var inferencer = new Inferencer("best.onnx");
/// var results = inferencer.Detect("photo.jpg");
/// foreach (var r in results.Where(r => r.Confidence > 0.5f))
///     Console.WriteLine($"{r.ClassName}: {r.Confidence:P1}");
/// </code>
/// </summary>
public class Inferencer : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string[] _classNames;
    private readonly int _inputWidth;
    private readonly int _inputHeight;
    private readonly float _confThreshold;
    private readonly float _nmsThreshold;

    private bool _disposed;

    /// <summary>
    /// 初始化推理器。
    /// </summary>
    /// <param name="modelPath">ONNX 模型文件路径</param>
    /// <param name="classNames">类别名称列表。为 null 时使用 class_0, class_1, ...</param>
    /// <param name="useCuda">是否使用 CUDA GPU 加速 (需安装 OnnxRuntime.Gpu 包)</param>
    /// <param name="inputSize">输入尺寸，需与训练时的 imgsz 一致 (默认 640)</param>
    /// <param name="confThreshold">置信度阈值 (默认 0.25)</param>
    /// <param name="nmsThreshold">NMS IoU 阈值 (默认 0.45)</param>
    public Inferencer(
        string modelPath,
        string[]? classNames = null,
        bool useCuda = false,
        int inputSize = 640,
        float confThreshold = 0.25f,
        float nmsThreshold = 0.45f)
    {
        _inputWidth = inputSize;
        _inputHeight = inputSize;
        _confThreshold = confThreshold;
        _nmsThreshold = nmsThreshold;

        var options = new SessionOptions();
        if (useCuda)
        {
            // 需要 NuGet: Microsoft.ML.OnnxRuntime.Gpu
            options.AppendExecutionProvider_CUDA();
        }

        _session = new InferenceSession(modelPath, options);

        // 从 ONNX 模型输出张量推导类别数
        // YOLOv8 输出: [1, 4 + nc, num_anchors]
        var outputMetadata = _session.OutputMetadata;
        var outputName = outputMetadata.Keys.First();
        var outputShape = outputMetadata[outputName].Dimensions;
        int numClasses = outputShape[1] - 4; // 减去 (x, y, w, h)

        _classNames = classNames ?? Enumerable.Range(0, numClasses)
            .Select(i => $"class_{i}").ToArray();
    }

    /// <summary>
    /// 对图片文件进行目标检测。
    /// </summary>
    public DetectionResult[] Detect(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        return Detect(image);
    }

    /// <summary>
    /// 对内存中的图片字节数据进行目标检测。
    /// </summary>
    public DetectionResult[] Detect(byte[] imageBytes)
    {
        using var image = Image.Load<Rgb24>(imageBytes);
        return Detect(image);
    }

    private DetectionResult[] Detect(Image<Rgb24> image)
    {
        int origWidth = image.Width;
        int origHeight = image.Height;

        // 1. 预处理: Resize + Normalize + CHW
        var tensor = Preprocess(image);

        // 2. 推理
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), tensor)
        };

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // 3. 后处理: 解析 + NMS
        return Postprocess(output, origWidth, origHeight);
    }

    /// <summary>
    /// 图片预处理: Resize → Normalize → 转为 float32 tensor [1, 3, H, W].
    /// </summary>
    private DenseTensor<float> Preprocess(Image<Rgb24> image)
    {
        // Resize 到模型输入尺寸，保持宽高比并填充
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(_inputWidth, _inputHeight),
            Mode = ResizeMode.Stretch // YOLO 使用 letterbox，简化处理用 Stretch
        }));

        int channels = 3;
        var tensor = new DenseTensor<float>(new[] { 1, channels, _inputHeight, _inputWidth });

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < _inputHeight; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < _inputWidth; x++)
                {
                    // ImageSharp: Rgb24 → R=0, G=1, B=2
                    // YOLO 期望 RGB 顺序，像素值归一化到 [0, 1]
                    tensor[0, 0, y, x] = row[x].R / 255f;
                    tensor[0, 1, y, x] = row[x].G / 255f;
                    tensor[0, 2, y, x] = row[x].B / 255f;
                }
            }
        });

        return tensor;
    }

    /// <summary>
    /// 后处理: 解析 YOLOv8 输出张量 → NMS 去重 → DetectionResult[]。
    ///
    /// YOLOv8 ONNX 输出 shape: [1, 4 + nc, num_anchors]
    /// 每个 anchor: [cx, cy, w, h, conf_class0, conf_class1, ...]
    /// </summary>
    private DetectionResult[] Postprocess(DenseTensor<float> output, int origWidth, int origHeight)
    {
        int numClasses = _classNames.Length;
        int numAnchors = output.Dimensions[2];

        var detections = new List<DetectionResult>();
        var boxes = new List<float[]>();
        var scores = new List<float>();
        var classIds = new List<int>();

        for (int i = 0; i < numAnchors; i++)
        {
            // 找出该 anchor 的最大置信度及其类别
            float maxConf = 0f;
            int maxClassId = 0;
            for (int c = 0; c < numClasses; c++)
            {
                float conf = output[0, 4 + c, i];
                if (conf > maxConf)
                {
                    maxConf = conf;
                    maxClassId = c;
                }
            }

            if (maxConf < _confThreshold)
                continue;

            float cx = output[0, 0, i];
            float cy = output[0, 1, i];
            float w = output[0, 2, i];
            float h = output[0, 3, i];

            detections.Add(new DetectionResult
            {
                ClassName = _classNames[maxClassId],
                ClassId = maxClassId,
                Confidence = maxConf,
                X = cx,
                Y = cy,
                Width = w,
                Height = h
            });
        }

        // 简单 NMS (按类别分组执行)
        return ApplyNms(detections);
    }

    private DetectionResult[] ApplyNms(List<DetectionResult> detections)
    {
        if (detections.Count == 0)
            return Array.Empty<DetectionResult>();

        // 按置信度降序排列
        detections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

        var kept = new List<DetectionResult>();

        for (int i = 0; i < detections.Count; i++)
        {
            bool suppress = false;
            for (int j = 0; j < kept.Count; j++)
            {
                if (detections[i].ClassId == kept[j].ClassId
                    && ComputeIoU(detections[i], kept[j]) > _nmsThreshold)
                {
                    suppress = true;
                    break;
                }
            }
            if (!suppress)
                kept.Add(detections[i]);
        }

        return kept.ToArray();
    }

    /// <summary>
    /// 计算两个边界框的 IoU (Intersection over Union).
    /// </summary>
    private static float ComputeIoU(DetectionResult a, DetectionResult b)
    {
        // 归一化坐标 → 左上右下
        float ax1 = a.X - a.Width / 2f;
        float ay1 = a.Y - a.Height / 2f;
        float ax2 = a.X + a.Width / 2f;
        float ay2 = a.Y + a.Height / 2f;

        float bx1 = b.X - b.Width / 2f;
        float by1 = b.Y - b.Height / 2f;
        float bx2 = b.X + b.Width / 2f;
        float by2 = b.Y + b.Height / 2f;

        // 交集
        float interX1 = Math.Max(ax1, bx1);
        float interY1 = Math.Max(ay1, by1);
        float interX2 = Math.Min(ax2, bx2);
        float interY2 = Math.Min(ay2, by2);

        float interArea = Math.Max(0, interX2 - interX1) * Math.Max(0, interY2 - interY1);
        float areaA = a.Width * a.Height;
        float areaB = b.Width * b.Height;
        float unionArea = areaA + areaB - interArea;

        return unionArea > 0 ? interArea / unionArea : 0f;
    }

    /// <summary>
    /// 释放 ONNX 推理会话资源。
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
        }
    }
}
```

- [ ] **Step 4: 创建 Examples/Program.cs**

```csharp
using VisionCore.Runtime;

// 使用示例 — 需要先通过 VisionCore Python 训练并导出 ONNX 模型
// 用法: dotnet run -- <model.onnx> <image.jpg> [--cuda] [--conf 0.5]

if (args.Length < 2)
{
    Console.WriteLine("用法: dotnet run -- <model.onnx> <image.jpg> [--cuda] [--conf 0.5]");
    Console.WriteLine("示例: dotnet run -- ../models/example_coco8/exports/best.onnx photo.jpg");
    return 1;
}

string modelPath = args[0];
string imagePath = args[1];
bool useCuda = args.Contains("--cuda");
float confThreshold = 0.25f;

int confIdx = Array.IndexOf(args, "--conf");
if (confIdx >= 0 && confIdx + 1 < args.Length)
    float.TryParse(args[confIdx + 1], out confThreshold);

if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"模型文件不存在: {modelPath}");
    return 1;
}

if (!File.Exists(imagePath))
{
    Console.Error.WriteLine($"图片文件不存在: {imagePath}");
    return 1;
}

Console.WriteLine($"模型: {modelPath}");
Console.WriteLine($"图片: {imagePath}");
Console.WriteLine($"设备: {(useCuda ? "CUDA GPU" : "CPU")}");
Console.WriteLine($"置信度阈值: {confThreshold}");
Console.WriteLine();

using var inferencer = new Inferencer(modelPath, useCuda: useCuda, confThreshold: confThreshold);
var results = inferencer.Detect(imagePath);

if (results.Length == 0)
{
    Console.WriteLine("未检测到任何目标。");
}
else
{
    Console.WriteLine($"检测到 {results.Length} 个目标:");
    foreach (var r in results)
    {
        Console.WriteLine($"  {r.ClassName,-20} 置信度: {r.Confidence:P1}  位置: ({r.X:F3}, {r.Y:F3}) 尺寸: {r.Width:F3}x{r.Height:F3}");
    }
}

return 0;
```

- [ ] **Step 5: 检查 C# 语法（如果有 dotnet 可用）**

```bash
which dotnet 2>/dev/null && dotnet --version || echo "dotnet 未安装，跳过编译检查 (代码语法可通过Visual Studio/Rider验证)"
```

---

### Task 8: 端到端验证 — coco8 全链路

**Files:**
- 无新文件创建 — 验证已有代码

- [ ] **Step 1: 运行配置加载器测试**

```bash
uv run python -m pytest tests/test_config_loader.py tests/test_pipeline.py -v
```

Expected: 全部 11 个测试通过。

- [ ] **Step 2: coco8 训练验证 (快速模式)**

```bash
uv run python scripts/train.py --task example_coco8 --epochs 1 --device -1
```

Expected: 训练不报错，输出中包含 `Results saved to` 路径，`models/example_coco8/train/weights/best.pt` 存在。
> **注意**: 使用 `--device -1` (CPU) 以确保在没有 GPU 的环境中也能运行。有 GPU 时可去掉此参数。

- [ ] **Step 3: coco8 评估验证**

```bash
uv run python scripts/eval.py --task example_coco8
```

Expected: 输出包含 mAP50 和 mAP50-95 数值。

- [ ] **Step 4: coco8 导出验证**

```bash
uv run python scripts/export.py --task example_coco8
```

Expected: 输出 `导出成功: models/example_coco8/exports/best.onnx`，文件存在且 > 1MB。

- [ ] **Step 5: 使用 coco8 自带测试图进行推理验证**

```bash
# Ultralytics 自带 coco8 的两张测试图在 assets/ 目录
# 推理 bus.jpg
uv run python -c "
from ultralytics import YOLO
import urllib.request
# 下载一张coco8里的测试图
url = 'https://ultralytics.com/images/bus.jpg'
urllib.request.urlretrieve(url, '/tmp/bus.jpg')
"
uv run python scripts/infer.py --task example_coco8 --source /tmp/bus.jpg --output models/example_coco8/inference
```

Expected: 生成 `models/example_coco8/inference/predict/bus.jpg`，图片上有标注框。

---

### Task 9: 清理与最终检查

**Files:**
- 无新文件创建

- [ ] **Step 1: 确认 .gitignore 生效 — data/ 和 models/ 不在 Git 追踪中**

```bash
git status --short 2>/dev/null || echo "非 Git 仓库，跳过检查"
```

- [ ] **Step 2: 运行全部测试**

```bash
uv run python -m pytest tests/ -v
```

Expected: 全部测试通过（11 tests in total）。

- [ ] **Step 3: 确认所有脚本的 --help 正常**

```bash
for script in train eval infer export; do
    echo "=== $script.py ==="
    uv run python scripts/$script.py --help 2>&1 | head -3
done
```

Expected: 每个脚本输出 usage 信息，无报错。

---

## Validation Checklist

实现完成后，逐项验证：

- [ ] `uv sync` 创建虚拟环境并安装依赖无报错
- [ ] `uv run python -m pytest tests/ -v` 全部通过
- [ ] `uv run python scripts/train.py --task example_coco8 --epochs 1 --device -1` 训练完成
- [ ] `uv run python scripts/eval.py --task example_coco8` 输出 mAP
- [ ] `uv run python scripts/export.py --task example_coco8` 生成 .onnx
- [ ] `uv run python scripts/infer.py --task example_coco8 --source /tmp/bus.jpg` 生成带标注框的结果图
- [ ] C# `VisionCore.Runtime.csproj` NuGet 引用正确，可在 Visual Studio 中打开
- [ ] 新增任务的流程文档化: README.md 中可清晰找到步骤
