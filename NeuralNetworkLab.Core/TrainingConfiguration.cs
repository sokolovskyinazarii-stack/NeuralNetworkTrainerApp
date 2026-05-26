namespace NeuralNetworkLab.Core;

public sealed class TrainingConfiguration
{
    public int TrainSize { get; set; } = 5000;
    public int TestSize { get; set; } = 2000;
    public int HiddenNeurons { get; set; } = 128;
    public int Epochs { get; set; } = 5;
    public int BatchSize { get; set; } = 32;
    public double LearningRate { get; set; } = 0.015;
    public bool UseSecondHiddenLayer { get; set; }
    public int SecondHiddenNeurons { get; set; } = 64;
    public int Seed { get; set; } = Random.Shared.Next();
}
