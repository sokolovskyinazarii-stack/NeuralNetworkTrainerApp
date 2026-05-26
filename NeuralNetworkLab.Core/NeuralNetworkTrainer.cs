namespace NeuralNetworkLab.Core;

public sealed class NeuralNetworkTrainer
{
    public TrainingResult Run(TrainingConfiguration config, IProgress<string>? progress = null)
    {
        var dataset = new SyntheticDigitDataset(config.Seed);
        var train = dataset.Create(config.TrainSize).ToList();
        var test = dataset.Create(config.TestSize).ToList();

        var layers = config.UseSecondHiddenLayer
            ? new[] { 784, config.HiddenNeurons, config.SecondHiddenNeurons, 10 }
            : new[] { 784, config.HiddenNeurons, 10 };

        var network = new DenseNeuralNetwork(layers, config.Seed);
        var history = new List<EpochMetric>();

        progress?.Report($"Архітектура мережі: {network.Architecture}");
        progress?.Report($"Seed генератора: {config.Seed}");
        progress?.Report($"Навчальна вибірка: {train.Count}; тестова вибірка: {test.Count}");

        for (int epoch = 1; epoch <= config.Epochs; epoch++)
        {
            Shuffle(train, config.Seed + epoch);
            double loss = network.TrainEpoch(train, config.LearningRate);
            double trainAcc = network.Accuracy(train.Take(Math.Min(1000, train.Count)).ToList(), out _);
            double testAcc = network.Accuracy(test, out _);

            history.Add(new EpochMetric
            {
                Epoch = epoch,
                Loss = loss,
                TrainAccuracy = trainAcc,
                TestAccuracy = testAcc
            });

            progress?.Report($"Епоха {epoch}/{config.Epochs}: loss={loss:F4}; train={trainAcc:F2}%; test={testAcc:F2}%");
        }

        double accuracy = network.Accuracy(test, out int correct);
        var predictions = test.Take(10)
            .Select((sample, index) =>
            {
                int predicted = network.Predict(sample.Pixels, out double confidence);
                return new PredictionRow
                {
                    Index = index,
                    Expected = sample.Label,
                    Predicted = predicted,
                    Confidence = confidence
                };
            })
            .ToList();

        DigitSample? mistake = null;
        int mistakePredicted = -1;
        double mistakeConfidence = 0;

        foreach (var sample in test)
        {
            int predicted = network.Predict(sample.Pixels, out double confidence);
            if (predicted == sample.Label) continue;
            mistake = sample;
            mistakePredicted = predicted;
            mistakeConfidence = confidence;
            break;
        }

        return new TrainingResult
        {
            Configuration = config,
            Architecture = network.Architecture,
            Accuracy = accuracy,
            Correct = correct,
            Total = test.Count,
            History = history,
            FirstPredictions = predictions,
            FirstMistake = mistake,
            FirstMistakePredicted = mistakePredicted,
            FirstMistakeConfidence = mistakeConfidence
        };
    }

    private static void Shuffle<T>(IList<T> list, int seed)
    {
        var random = new Random(seed);
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
