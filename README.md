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
