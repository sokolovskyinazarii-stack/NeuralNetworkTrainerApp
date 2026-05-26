using NeuralNetworkLab.Core;

var samples = new SyntheticDigitDataset(42)
    .Create(300)
    .Where(sample => sample.Label == 0)
    .Take(20)
    .ToList();

Assert(samples.Count > 0, "Dataset must include zero samples.");

foreach (var sample in samples)
{
    double center = Average(sample.Pixels, 11, 10, 6, 8);
    double top = Average(sample.Pixels, 5, 2, 18, 7);
    double bottom = Average(sample.Pixels, 5, 19, 18, 7);
    double left = Average(sample.Pixels, 3, 6, 7, 18);
    double right = Average(sample.Pixels, 18, 6, 7, 18);

    Assert(center < 0.14, $"Zero center should stay empty, got {center:F3}.");
    Assert(top > 0.10, $"Zero top stroke is too weak, got {top:F3}.");
    Assert(bottom > 0.10, $"Zero bottom stroke is too weak, got {bottom:F3}.");
    Assert(left > 0.10, $"Zero left stroke is too weak, got {left:F3}.");
    Assert(right > 0.10, $"Zero right stroke is too weak, got {right:F3}.");
}

var defaultTraining = new NeuralNetworkTrainer().Run(new TrainingConfiguration
{
    TrainSize = 5000,
    TestSize = 2000,
    HiddenNeurons = 128,
    Epochs = 5,
    Seed = 42
});

Assert(
    defaultTraining.Accuracy >= 45 && defaultTraining.Accuracy <= 60,
    $"Default training should be close to 50%, got {defaultTraining.Accuracy:F2}%.");

Console.WriteLine("Synthetic digit dataset checks passed.");

static double Average(double[] pixels, int startX, int startY, int width, int height)
{
    double sum = 0;
    int count = 0;

    for (int y = startY; y < startY + height; y++)
    {
        for (int x = startX; x < startX + width; x++)
        {
            sum += pixels[y * 28 + x];
            count++;
        }
    }

    return sum / count;
}

static void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}
