namespace NeuralNetworkLab.Core;

public sealed class EpochMetric
{
    public int Epoch { get; init; }
    public double TrainAccuracy { get; init; }
    public double TestAccuracy { get; init; }
    public double Loss { get; init; }
}

public sealed class PredictionRow
{
    public int Index { get; init; }
    public int Expected { get; init; }
    public int Predicted { get; init; }
    public double Confidence { get; init; }
}

public sealed class TrainingResult
{
    public TrainingConfiguration Configuration { get; init; } = new();
    public string Architecture { get; init; } = string.Empty;
    public double Accuracy { get; init; }
    public int Correct { get; init; }
    public int Total { get; init; }
    public IReadOnlyList<EpochMetric> History { get; init; } = Array.Empty<EpochMetric>();
    public IReadOnlyList<PredictionRow> FirstPredictions { get; init; } = Array.Empty<PredictionRow>();
    public DigitSample? FirstMistake { get; init; }
    public int FirstMistakePredicted { get; init; }
    public double FirstMistakeConfidence { get; init; }
}
