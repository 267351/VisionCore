# deploy/VisionCore.Runtime/ — C# ONNX Inferencer

## OVERVIEW
.NET 8.0 class library for YOLOv8 ONNX inference. Separate project from Python training — linked only by `.onnx` file exchange. Cross-platform (Windows/Linux) via ImageSharp instead of System.Drawing.

## STRUCTURE
```
deploy/VisionCore.Runtime/
├── VisionCore.Runtime.csproj   # net8.0, OnnxRuntime + ImageSharp
├── Inferencer.cs               # Load ONNX → preprocess → infer → NMS
├── Models/
│   └── DetectionResult.cs      # Immutable record, normalized coords + ToPixel()
└── Examples/
    └── Program.cs              # CLI consumer with manual arg parsing
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Load model and detect | `Inferencer.cs:124-170` | Constructor loads ONNX; `Detect(string/byte[])` overloads |
| Preprocessing | `Inferencer.cs:178-210` | Stretch-resize → /255 normalize → CHW DenseTensor |
| Postprocessing | `Inferencer.cs:218-259` | Conf filter + class-aware greedy NMS |
| Bbox utility | `DetectionResult.cs:35-44` | `ToPixel()` — normalized → clamped pixel rect |
| Usage example | `Examples/Program.cs` | `dotnet run -- <model.onnx> <image.jpg> [--cuda]` |

## CONVENTIONS

- **Normalized coords**: All bbox values in `DetectionResult` are [0,1] — use `ToPixel()` to convert
- **Class derivation**: Inferencer auto-derives `numClasses` from ONNX output shape `[1, 4+nc, anchors]`
- **NMS**: Class-aware greedy — sort by confidence desc, suppress per-class by IoU > 0.45
- **Dispose pattern**: `Inferencer : IDisposable` — always wrap in `using var`

## GOTCHAS

- **Stretch vs Letterbox**: Preprocess uses `ResizeMode.Stretch`, NOT YOLO letterbox. On non-square images, bbox positions will be misaligned vs. Python inference results. Only accurate on square images.
- **Bbox rescale missing**: Detected coordinates stay in input-tensor space — not rescaled to original image dimensions. This is consistent with the documented design but differs from typical YOLO postprocessing.
- **GPU support**: Default is CPU. To use CUDA, switch NuGet from `Microsoft.ML.OnnxRuntime` → `Microsoft.ML.OnnxRuntime.Gpu` and pass `useCuda: true`.

## ANTI-PATTERNS

- **DO NOT** use `System.Drawing` — breaks Linux compatibility; use `SixLabors.ImageSharp`
- **DO NOT** modify detection coordinates after postprocess without updating `ToPixel()` logic
