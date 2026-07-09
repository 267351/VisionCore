# VisionCore — 焊接视觉 AI 训练平台 设计文档

**日期**: 2026-07-09
**作者**: Sisyphus + 用户协作
**状态**: 已确认，待实施

---

## 1. 项目目标

构建一个**平台化、配置驱动的物体识别模型训练框架**，覆盖焊接领域的全部视觉识别需求。核心原则：

- **新增识别任务 = 新建配置文件 + 放入数据**，无需改代码
- **训练和推理分离**：Python 做训练，C# 做推理部署
- **ONNX 作为桥接格式**，跨 Windows/Linux 部署

## 2. 技术选型

| 层面 | 选择 | 理由 |
|------|------|------|
| 训练框架 | Ultralytics YOLOv8 | 精度高、上手快、部署友好 |
| 模型系列 | YOLOv8n/s/m/l/x | 支持 detect/classify/segment 三种任务 |
| Python 包管理 | uv | 速度快、类似 npm，C# 工程师友好 |
| 推理引擎 | ONNX Runtime | 跨平台，C# NuGet 直接调用 |
| C# 运行时 | .NET 8.0 | 跨 Windows/Linux |
| 标注格式 | YOLO 格式 (.txt) | YOLOv8 原生支持 |
| 训练硬件 | 本地 NVIDIA GPU (CUDA) | 用户已有 |
| 部署平台 | Windows + Linux 双平台 | 产线需求 |

### Python 依赖

```
ultralytics>=8.2.0
opencv-python>=4.9.0
pyyaml>=6.0
onnx>=1.16.0
onnxruntime-gpu>=1.18.0  # 训练机
pytest>=8.0               # 测试
```

### C# NuGet 依赖

```
Microsoft.ML.OnnxRuntime.Gpu  # 或 .Cpu for CPU-only
Microsoft.ML.OnnxRuntime.Extensions
SixLabors.ImageSharp          # 图片预处理
```

## 3. 项目结构

```
VisionCore/
├── configs/                          # 配置层
│   ├── default.yaml                  # 全局默认参数
│   └── tasks/                        # 每个识别任务一个文件
│       ├── example_coco8.yaml        # 框架验证用
│       ├── weld_defect.yaml          # 焊缝缺陷检测
│       ├── rod_type.yaml             # 焊条型号识别
│       └── rod_state.yaml            # 焊条头状态识别
│
├── data/                             # 数据集 (.gitignore)
│   └── {task_name}/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── models/                           # 训练产出 (.gitignore)
│   └── {task_name}/
│       ├── weights/
│       │   └── best.pt
│       ├── exports/
│       │   └── best.onnx
│       └── runs/
│
├── scripts/                          # CLI 层 (Python)
│   ├── train.py                      # 训练入口
│   ├── eval.py                       # 评估入口
│   ├── infer.py                      # 推理入口
│   ├── export.py                     # 导出入口
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py          # 配置合并
│       └── logger.py                 # 统一日志
│
├── deploy/                           # C# 推理运行时
│   └── VisionCore.Runtime/
│       ├── VisionCore.Runtime.csproj
│       ├── Inferencer.cs
│       ├── Models/
│       │   └── DetectionResult.cs
│       └── Examples/
│           └── Program.cs
│
├── tests/                            # 测试
│   ├── __init__.py
│   ├── test_config_loader.py
│   └── test_pipeline.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 4. 配置系统设计

### 4.1 全局默认配置 `configs/default.yaml`

```yaml
model:
  size: "n"          # n/s/m/l/x
  pretrained: true

train:
  epochs: 100
  batch: 16
  imgsz: 640
  device: 0
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

### 4.2 任务配置示例 `configs/tasks/weld_defect.yaml`

```yaml
task:
  name: "weld_defect"
  type: "detect"           # detect | classify | segment
  description: "焊缝缺陷检测"

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

### 4.3 配置合并规则

- `task.yaml` 中定义的字段覆盖 `default.yaml` 同名字段
- ⚠️ **浅合并**：task 文件只需写要覆盖的字段，其余沿用默认值
- 配置错误（文件不存在、必填字段缺失）在脚本入口处检查，提前报错

### 4.4 新增识别任务流程

1. 数据放入 `data/{task_name}/images/{train,val}/`
2. 标注产物放入 `data/{task_name}/labels/{train,val}/`（YOLO 格式）
3. 新建 `configs/tasks/{task_name}.yaml`
4. `python scripts/train.py --task {task_name}`

> **特殊处理**：对于 Ultralytics 内置数据集（如 `coco8.yaml`），`data.path` 直接填写数据集名。脚本检测到路径不含 `data/` 前缀时，透传给 YOLO API 而非做本地文件验证。

## 5. CLI 脚本接口

### 5.1 `train.py`

```bash
python scripts/train.py --task <name>
                        [--model-size n|s|m|l|x]
                        [--epochs N] [--batch N] [--imgsz N]
                        [--device N] [--resume]
```

流程：加载配置 → CLI参数覆盖 → 创建输出目录 → 训练 → 自动保存 best.pt → 打印 mAP

### 5.2 `eval.py`

```bash
python scripts/eval.py --task <name>
                       [--model-size n|s|m|l|x]
```

输出：mAP50、mAP50-95、每类 Precision/Recall

### 5.3 `infer.py`

```bash
python scripts/infer.py --task <name> --source <path|0>
                        [--model-size n|s|m|l|x]
                        [--output <dir>]
                        [--conf 0.25]
```

`--source` 支持：图片路径、目录路径、摄像头编号(0)

### 5.4 `export.py`

```bash
python scripts/export.py --task <name>
                         [--model-size n|s|m|l|x]
                         [--format onnx|engine|tflite]
                         [--half] [--dynamic]
```

默认导出 ONNX 到 `models/{task}/exports/best.onnx`

## 6. C# 推理运行时

### 6.1 核心类接口

```csharp
namespace VisionCore.Runtime;

public class Inferencer : IDisposable
{
    public Inferencer(string modelPath, bool useCuda = false);
    public DetectionResult[] Detect(string imagePath);
    public DetectionResult[] Detect(byte[] imageBytes);
    public void Dispose();
}

public record DetectionResult
{
    public string ClassName { get; init; }
    public int ClassId { get; init; }
    public float Confidence { get; init; }
    public float X { get; init; }      // 中心x，归一化0-1
    public float Y { get; init; }      // 中心y，归一化0-1
    public float Width { get; init; }  // 归一化0-1
    public float Height { get; init; } // 归一化0-1
}
```

### 6.2 预处理对齐

C# 侧图片预处理必须与 Python 训练时的预处理参数一致：
- 输入尺寸：与 `configs/default.yaml` 的 `train.imgsz` 保持一致（默认 640）
- 色彩空间：BGR → RGB
- 归一化：像素值 / 255.0
- 格式：CHW (1, 3, H, W)，float32

### 6.3 后处理

- ONNX 输出张量 shape: `[1, 84, 8400]`（YOLOv8n, 640x640, 80类时）
- 实际类别数由 ONNX 模型决定，从输出张量尺寸推导
- NMS 阈值默认 0.45
- 置信度阈值由调用方指定，默认 0.25

## 7. 验证策略

### 7.1 框架验证（example_coco8 任务）

用 Ultralytics 内置的 coco8 数据集验证框架完整链路：
1. `python scripts/train.py --task example_coco8`  → 训练不报错，生成 best.pt
2. `python scripts/eval.py --task example_coco8`   → 输出 mAP 数值
3. `python scripts/infer.py --task example_coco8 --source <test_img>` → 生成标注图
4. `python scripts/export.py --task example_coco8` → 生成 best.onnx

### 7.2 单元测试

- `test_config_loader.py`：合并逻辑、缺失文件报错、字段覆盖
- `test_pipeline.py`：验证 train→eval→export 链路不中断

### 7.3 部署验证

C# Inferencer 加载 coco8 导出的 ONNX → 推理一张图 → 结果与 Python infer.py 结果一致

## 8. 未来扩展点（不在本期范围）

- MLflow 模型版本管理
- 多后端适配（RT-DETR、YOLO-World）
- Blazor Web 界面
- 数据集版本管理
- 自动标注辅助
