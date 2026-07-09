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
