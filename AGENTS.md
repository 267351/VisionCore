# VisionCore — Project Knowledge Base

**Generated:** 2026-07-10
**Commit:** 2b2f4b0
**Branch:** dev

## OVERVIEW
Config-driven YOLOv8 training platform for welding vision AI. Python CLI trains models; C# ONNX Runtime deploys them. One YAML per task, zero code changes to add a new detection job.

## STRUCTURE
```
VisionCore/
├── configs/tasks/     # Per-task YAML (override default.yaml)
├── scripts/           # CLI: train.py, eval.py, infer.py, export.py
│   └── utils/         # config_loader, logger
├── deploy/Runtime/    # C# ONNX inferencer (separate project)
├── tests/             # pytest — 13 tests, all pass
└── docs/              # Tutorial + specs
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a new detection model | `configs/tasks/` | Create `<task>.yaml`, put data in `data/<task>/` |
| Change default training params | `configs/default.yaml` | Affects all tasks unless overridden |
| Debug training/eval logic | `scripts/train.py` | All 4 scripts share the same pattern |
| Debug config loading | `scripts/utils/config_loader.py` | `_deep_merge` + `_validate_task_config` |
| Use model in C# app | `deploy/VisionCore.Runtime/Inferencer.cs` | Load .onnx → Detect() → DetectionResult[] |
| Run tests | `uv run python -m pytest tests/ -v` | Or `source .venv/bin/activate && pytest` |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `load_config(task_name)` | fn | scripts/utils/config_loader.py:48 | 7 callers | Merges default.yaml + task.yaml |
| `get_logger(name)` | fn | scripts/utils/logger.py:6 | 6 callers | Stdout logger with timestamps |
| `_deep_merge(base, override)` | fn | scripts/utils/config_loader.py:17 | 1 (internal) | Recursive dict merge |
| `Inferencer` | class | deploy/.../Inferencer.cs:24 | C# | ONNX load → preprocess → infer → NMS |
| `DetectionResult` | record | deploy/.../DetectionResult.cs:11 | C# | Normalized bbox + ToPixel() |

## CONVENTIONS

- **Task naming**: snake_case, matches `data/<name>/` dir AND `configs/tasks/<name>.yaml`
- **Config merge**: shallow override — task.yaml fields replace default.yaml; nested dicts merge recursively
- **CLI pattern**: Every script: `argparse` → `load_config(task)` → CLI args override config → YOLO call
- **Logging**: All scripts use `get_logger(__name__)` from utils/logger, INFO level to stdout
- **Type hints**: 100% coverage on production code; test code omits return types (standard pytest practice)
- **Imports**: Scripts insert `sys.path` to import from utils/; tests use `sys.path.insert` (redundant with `pythonpath = ["scripts"]` in pyproject.toml)
- **Builtin datasets**: When `data.path` doesn't start with `data/`, it's treated as Ultralytics builtin name (e.g. `coco8.yaml`)
- **Python version**: `>=3.10` (minimum); actual runtime 3.13
- **No `uv`**: Use pip + venv despite pyproject.toml existing; the tutorial documents pip flow

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** edit `scripts/utils/config_loader.py` without updating tests in `tests/test_config_loader.py`
- **DO NOT** add hardcoded task names — always go through config files
- **DO NOT** add `sys.path.insert` hacks in test files — `pythonpath=["scripts"]` in pyproject.toml already handles it
- **NEVER** commit `data/`, `models/`, `*.pt`, `*.onnx`, `runs/`, `datasets/` — all in .gitignore
- **NEVER** mix detect/classify/segment task types in one config — each task is one type

## COMMANDS
```bash
# Setup (once)
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install ultralytics opencv-python pyyaml onnx onnxruntime pytest

# Test
pytest tests/ -v                                     # 13 tests

# Train / Eval / Export / Infer (coco8 smoke test)
python scripts/train.py --task example_coco8 --epochs 1
python scripts/eval.py --task example_coco8
python scripts/export.py --task example_coco8
python scripts/infer.py --task example_coco8 --source test.jpg
```

## NOTES

- **C# runtime gotcha**: `Inferencer.Preprocess()` uses `ResizeMode.Stretch` (not YOLO letterbox) — bbox coordinates are in input-tensor space, not rescaled to original image dims. Boxes will be misaligned on non-square images.
- **Config validation**: Missing `task`, `task.name`, `task.type`, or `data` fields → `KeyError` at load time. Malformed YAML → `yaml.YAMLError`.
- **Test gap**: No tests for `train.py`/`eval.py`/`export.py` script entry-points; no C# tests; `temp_dir` fixture is defined but unused.
