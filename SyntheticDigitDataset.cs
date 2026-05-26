namespace NeuralNetworkLab.Core;

public sealed class SyntheticDigitDataset
{
    private readonly Random _random;
    private static readonly Segment Top = new(new Point2(8, 5), new Point2(20, 5));
    private static readonly Segment UpperLeft = new(new Point2(7, 6), new Point2(7, 14));
    private static readonly Segment UpperRight = new(new Point2(21, 6), new Point2(21, 14));
    private static readonly Segment Middle = new(new Point2(8, 14), new Point2(20, 14));
    private static readonly Segment LowerLeft = new(new Point2(7, 14), new Point2(7, 22));
    private static readonly Segment LowerRight = new(new Point2(21, 14), new Point2(21, 22));
    private static readonly Segment Bottom = new(new Point2(8, 23), new Point2(20, 23));
    private static readonly Segment CenterStem = new(new Point2(14, 5), new Point2(14, 22));
    private static readonly Segment SmallBase = new(new Point2(10, 23), new Point2(18, 23));
    private static readonly Segment[][] Digits =
    {
        new[] { Top, UpperLeft, UpperRight, LowerLeft, LowerRight, Bottom },
        new[] { CenterStem, SmallBase },
        new[] { Top, UpperRight, Middle, LowerLeft, Bottom },
        new[] { Top, UpperRight, Middle, LowerRight, Bottom },
        new[] { UpperLeft, UpperRight, Middle, LowerRight },
        new[] { Top, UpperLeft, Middle, LowerRight, Bottom },
        new[] { Top, UpperLeft, Middle, LowerLeft, LowerRight, Bottom },
        new[] { Top, UpperRight, LowerRight },
        new[] { Top, UpperLeft, UpperRight, Middle, LowerLeft, LowerRight, Bottom },
        new[] { Top, UpperLeft, UpperRight, Middle, LowerRight, Bottom }
    };

    public SyntheticDigitDataset(int seed = 42)
    {
        _random = new Random(seed);
    }

    public IReadOnlyList<DigitSample> Create(int count)
    {
        var samples = new List<DigitSample>(count);
        for (int i = 0; i < count; i++)
        {
            int label = i % 10;
            samples.Add(new DigitSample(RenderDigit(label), label));
        }

        return samples.OrderBy(_ => _random.Next()).ToList();
    }

    private double[] RenderDigit(int label)
    {
        var pixels = new double[28 * 28];
        double difficulty = _random.NextDouble();
        bool veryHard = difficulty < 0.62;
        bool hard = difficulty < 0.93;

        if (hard)
        {
            DrawDistractorDigit(pixels, PickConfuser(label), veryHard ? 0.62 : 0.42);
        }

        double scaleX = 0.80 + _random.NextDouble() * 0.34;
        double scaleY = 0.80 + _random.NextDouble() * 0.34;
        double angle = (_random.NextDouble() - 0.5) * (veryHard ? 0.72 : hard ? 0.52 : 0.30);
        double offsetX = (_random.NextDouble() - 0.5) * (veryHard ? 7.4 : hard ? 5.6 : 3.8);
        double offsetY = (_random.NextDouble() - 0.5) * (veryHard ? 6.4 : hard ? 4.8 : 3.2);
        double baseThickness = (veryHard ? 1.2 : hard ? 1.45 : 1.8) + _random.NextDouble() * 1.45;
        double inkScale = veryHard ? 0.42 + _random.NextDouble() * 0.24 : hard ? 0.55 + _random.NextDouble() * 0.22 : 0.78 + _random.NextDouble() * 0.18;

        foreach (var segment in Digits[label])
        {
            if (label != 0 && ((veryHard && _random.NextDouble() < 0.24) || (hard && _random.NextDouble() < 0.08)))
            {
                continue;
            }

            var start = Transform(segment.Start, scaleX, scaleY, angle, offsetX, offsetY);
            var end = Transform(segment.End, scaleX, scaleY, angle, offsetX, offsetY);
            double jitter = label == 0 ? 0.55 : 1.05;
            start = Jitter(start, jitter);
            end = Jitter(end, jitter);

            var control = CreateControlPoint(start, end, label);
            double thickness = Math.Max(1.55, baseThickness + (_random.NextDouble() - 0.5) * 0.75);
            if (label == 0)
            {
                thickness = Math.Min(thickness, 2.55);
            }

            double dropout = label == 0
                ? 0.10 + _random.NextDouble() * 0.13
                : (veryHard ? 0.26 : hard ? 0.16 : 0.07) + _random.NextDouble() * (veryHard ? 0.24 : hard ? 0.20 : 0.10);
            DrawStroke(pixels, start, control, thickness, dropout, inkScale);
            DrawStroke(pixels, control, end, thickness, dropout, inkScale);
        }

        if (hard)
        {
            AddConfusingFragments(pixels, label, veryHard);
        }

        if (veryHard)
        {
            ApplyOcclusion(pixels);
        }

        ApplySpeckle(pixels);
        ApplySmoothing(pixels);

        for (int i = 0; i < pixels.Length; i++)
        {
            double noise = (_random.NextDouble() - 0.30) * (veryHard ? 0.24 : hard ? 0.18 : 0.10);
            pixels[i] = Math.Clamp(pixels[i] + noise, 0, 1);
        }

        if (label == 0)
        {
            PreserveZeroHole(pixels);
        }

        return pixels;
    }

    private static Point2 Transform(Point2 point, double scaleX, double scaleY, double angle, double offsetX, double offsetY)
    {
        const double center = 14;

        double x = (point.X - center) * scaleX;
        double y = (point.Y - center) * scaleY;
        double cos = Math.Cos(angle);
        double sin = Math.Sin(angle);

        return new Point2(
            x * cos - y * sin + center + offsetX,
            x * sin + y * cos + center + offsetY);
    }

    private static Point2 Midpoint(Point2 start, Point2 end)
    {
        return new Point2((start.X + end.X) / 2, (start.Y + end.Y) / 2);
    }

    private Point2 Jitter(Point2 point, double range)
    {
        return new Point2(
            point.X + (_random.NextDouble() - 0.5) * range,
            point.Y + (_random.NextDouble() - 0.5) * range);
    }

    private Point2 CreateControlPoint(Point2 start, Point2 end, int label)
    {
        var control = Midpoint(start, end);

        if (label == 0)
        {
            double dx = control.X - 14;
            double dy = control.Y - 14;
            double length = Math.Sqrt(dx * dx + dy * dy);
            if (length > 0.001)
            {
                control = new Point2(
                    control.X + dx / length * 1.25,
                    control.Y + dy / length * 1.25);
            }

            return Jitter(control, 0.65);
        }

        return new Point2(
            control.X + (_random.NextDouble() - 0.5) * 3.8,
            control.Y + (_random.NextDouble() - 0.5) * 3.8);
    }

    private void DrawStroke(double[] pixels, Point2 start, Point2 end, double thickness, double dropout, double inkScale)
    {
        double minX = Math.Min(start.X, end.X) - thickness - 2;
        double maxX = Math.Max(start.X, end.X) + thickness + 2;
        double minY = Math.Min(start.Y, end.Y) - thickness - 2;
        double maxY = Math.Max(start.Y, end.Y) + thickness + 2;

        for (int y = Math.Max(0, (int)Math.Floor(minY)); y <= Math.Min(27, (int)Math.Ceiling(maxY)); y++)
        {
            for (int x = Math.Max(0, (int)Math.Floor(minX)); x <= Math.Min(27, (int)Math.Ceiling(maxX)); x++)
            {
                if (_random.NextDouble() < dropout)
                {
                    continue;
                }

                double distance = DistanceToSegment(x + 0.5, y + 0.5, start, end);
                double intensity = StrokeIntensity(distance, thickness);
                if (intensity > 0)
                {
                    double ink = intensity * inkScale * (0.78 + _random.NextDouble() * 0.28);
                    Put(pixels, x, y, ink);
                }
            }
        }
    }

    private static double DistanceToSegment(double x, double y, Point2 start, Point2 end)
    {
        double dx = end.X - start.X;
        double dy = end.Y - start.Y;
        double lengthSquared = dx * dx + dy * dy;
        if (lengthSquared == 0)
        {
            return Math.Sqrt(Math.Pow(x - start.X, 2) + Math.Pow(y - start.Y, 2));
        }

        double t = ((x - start.X) * dx + (y - start.Y) * dy) / lengthSquared;
        t = Math.Clamp(t, 0, 1);

        double nearestX = start.X + t * dx;
        double nearestY = start.Y + t * dy;
        return Math.Sqrt(Math.Pow(x - nearestX, 2) + Math.Pow(y - nearestY, 2));
    }

    private static double StrokeIntensity(double distance, double thickness)
    {
        double radius = thickness / 2;
        if (distance <= radius)
        {
            return 1.0;
        }

        double fade = 1 - (distance - radius) / 1.2;
        return Math.Clamp(fade * 0.75, 0, 0.75);
    }

    private void ApplySpeckle(double[] pixels)
    {
        int dust = _random.Next(34, 76);
        for (int i = 0; i < dust; i++)
        {
            int x = _random.Next(2, 26);
            int y = _random.Next(2, 26);
            Put(pixels, x, y, 0.12 + _random.NextDouble() * 0.34);
        }
    }

    private int PickConfuser(int label)
    {
        int[][] confusers =
        {
            new[] { 8, 6, 9 },
            new[] { 7, 4 },
            new[] { 3, 7 },
            new[] { 2, 5, 9 },
            new[] { 1, 9 },
            new[] { 3, 6, 8 },
            new[] { 5, 8, 0 },
            new[] { 1, 2 },
            new[] { 0, 6, 9 },
            new[] { 3, 4, 8 }
        };

        var options = confusers[label];
        return options[_random.Next(options.Length)];
    }

    private void DrawDistractorDigit(double[] pixels, int label, double inkScale)
    {
        double scaleX = 0.76 + _random.NextDouble() * 0.38;
        double scaleY = 0.76 + _random.NextDouble() * 0.38;
        double angle = (_random.NextDouble() - 0.5) * 0.62;
        double offsetX = (_random.NextDouble() - 0.5) * 7.0;
        double offsetY = (_random.NextDouble() - 0.5) * 6.0;
        double thickness = 1.1 + _random.NextDouble() * 0.9;

        foreach (var segment in Digits[label].OrderBy(_ => _random.Next()).Take(_random.Next(3, Digits[label].Length + 1)))
        {
            var start = Jitter(Transform(segment.Start, scaleX, scaleY, angle, offsetX, offsetY), 1.4);
            var end = Jitter(Transform(segment.End, scaleX, scaleY, angle, offsetX, offsetY), 1.4);
            var control = CreateControlPoint(start, end, label);
            DrawStroke(pixels, start, control, thickness, 0.28, inkScale);
            DrawStroke(pixels, control, end, thickness, 0.28, inkScale);
        }
    }

    private void AddConfusingFragments(double[] pixels, int label, bool veryHard)
    {
        int confuser = PickConfuser(label);
        int fragmentCount = veryHard ? _random.Next(2, 5) : _random.Next(1, 3);
        double ink = veryHard ? 0.56 : 0.38;

        for (int i = 0; i < fragmentCount; i++)
        {
            var segment = Digits[confuser][_random.Next(Digits[confuser].Length)];
            var start = Jitter(segment.Start, 5.4);
            var end = Jitter(segment.End, 5.4);
            var control = CreateControlPoint(start, end, confuser);
            double thickness = 1.45 + _random.NextDouble() * 1.25;
            DrawStroke(pixels, start, control, thickness, 0.18, ink);
            DrawStroke(pixels, control, end, thickness, 0.18, ink);
        }
    }

    private void ApplyOcclusion(double[] pixels)
    {
        int count = _random.Next(2, 5);
        for (int i = 0; i < count; i++)
        {
            int startX = _random.Next(2, 20);
            int startY = _random.Next(2, 20);
            int width = _random.Next(4, 9);
            int height = _random.Next(3, 8);
            double fade = 0.04 + _random.NextDouble() * 0.28;

            for (int y = startY; y < Math.Min(28, startY + height); y++)
            {
                for (int x = startX; x < Math.Min(28, startX + width); x++)
                {
                    int index = y * 28 + x;
                    pixels[index] *= fade;
                }
            }
        }
    }

    private static void PreserveZeroHole(double[] pixels)
    {
        for (int y = 10; y < 18; y++)
        {
            for (int x = 11; x < 17; x++)
            {
                int index = y * 28 + x;
                pixels[index] = Math.Min(pixels[index], 0.08);
            }
        }
    }

    private static void ApplySmoothing(double[] pixels)
    {
        var copy = (double[])pixels.Clone();

        for (int y = 1; y < 27; y++)
        {
            for (int x = 1; x < 27; x++)
            {
                double sum =
                    copy[(y - 1) * 28 + x] * 0.08 +
                    copy[(y + 1) * 28 + x] * 0.08 +
                    copy[y * 28 + x - 1] * 0.08 +
                    copy[y * 28 + x + 1] * 0.08 +
                    copy[y * 28 + x] * 0.68;

                pixels[y * 28 + x] = Math.Clamp(sum, 0, 1);
            }
        }
    }

    private static void Put(double[] pixels, int x, int y, double value)
    {
        if (x < 0 || x >= 28 || y < 0 || y >= 28) return;
        int index = y * 28 + x;
        pixels[index] = Math.Clamp(Math.Max(pixels[index], value), 0, 1);
    }

    private readonly record struct Point2(double X, double Y);

    private readonly record struct Segment(Point2 Start, Point2 End);
}
