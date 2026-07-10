# VisionCore — 焊接视觉AI训练平台

配置驱动的YOLOv8训练框架，覆盖焊接领域全部视觉识别需求。

## 快速开始

```bash
# 安装依赖（只需一次）
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install ultralytics opencv-python pyyaml onnx onnxruntime

# 验证框架 — 用 coco8 数据集跑通全链路
python scripts/train.py --task example_coco8 --epochs 1
python scripts/eval.py --task example_coco8
python scripts/export.py --task example_coco8
```

## 已支持的任务配置

| 任务 | 类型 | 类别数 |
|------|------|--------|
| `weld_defect` — 焊缝缺陷检测 | detect | 5 |
| `rod_type` — 焊条型号识别 | classify | 5 |
| `rod_state` — 焊条头状态识别 | detect | 4 |
| `example_coco8` — 框架验证 | detect | 80 (内置) |

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

## 新增识别任务

1. 数据放入 `data/{task_name}/images/{train,val}/`
2. 标注放入 `data/{task_name}/labels/{train,val}/`
3. 新建 `configs/tasks/{task_name}.yaml`
4. `python scripts/train.py --task {task_name}`

详细教程见 [`docs/新手完全指南.md`](docs/新手完全指南.md)。
