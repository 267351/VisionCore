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
