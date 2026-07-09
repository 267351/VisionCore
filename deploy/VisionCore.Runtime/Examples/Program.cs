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
