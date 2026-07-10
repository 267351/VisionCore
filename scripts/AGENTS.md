# scripts/ — Python CLI & Utilities

## OVERVIEW
4 YOLOv8 CLI entry points + 2 shared utils. All scripts follow the same pattern: `argparse` → `load_config(task)` → CLI overrides → YOLO call.

## WHERE TO LOOK
| Task | Script | Key Function |
|------|--------|-------------|
| Train a model | `train.py` | `main()` — config merge + CLI override + `model.train()` |
| Evaluate mAP | `eval.py` | `main()` — loads best.pt → `model.val()` |
| Run inference | `infer.py` | `main()` — loads best.pt → `model.predict()` |
| Export to ONNX | `export.py` | `main()` — loads best.pt → `model.export()` |
| Config merge logic | `utils/config_loader.py` | `load_config()` — deep merges default + task YAML |
| Logging | `utils/logger.py` | `get_logger(name)` — stdout logger with timestamps |

## CONVENTIONS

- **Pattern**: Every script does `sys.path.insert(0, Path(__file__).parent)` then `from utils import ...`
- **Config flow**: `load_config(task)` → CLI args override `cfg["train"]["epochs"]` etc. → pass to YOLO
- **Model naming**: `yolov8{size}{suffix}.pt` where suffix is "" (detect), "-cls" (classify), "-seg" (segment)
- **Output dir**: `models/{task_name}/` — YOLO creates `train/weights/best.pt` inside
- **Error exit**: `sys.exit(1)` on missing weights files; `load_config` raises `FileNotFoundError`/`KeyError`
- **No side effects in utils**: `config_loader.py` and `logger.py` are pure functions with no global state

## DEPENDENCY GRAPH

```
train / eval / infer / export
  └── utils.config_loader
        └── utils.logger
              └── (stdlib only: logging + sys)
```

## ANTI-PATTERNS

- **DO NOT** add `sys.path` hacks in utils/ — they're imported from scripts/ where path is already set
- **DO NOT** call YOLO directly without `load_config()` — always go through config
- **DO NOT** bypass `get_logger()` — use it for consistent timestamped output
