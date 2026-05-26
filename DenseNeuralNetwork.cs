namespace NeuralNetworkLab.Core;

public sealed class DenseNeuralNetwork
{
    private readonly int[] _layers;
    private readonly double[][][] _weights;
    private readonly double[][] _biases;
    private readonly Random _random;

    public DenseNeuralNetwork(int[] layers, int seed = 42)
    {
        if (layers.Length < 2) throw new ArgumentException("Network must contain at least input and output layers.");
        _layers = layers;
        _random = new Random(seed);
        _weights = new double[layers.Length - 1][][];
        _biases = new double[layers.Length - 1][];
        Initialize();
    }

    public string Architecture => string.Join(" -> ", _layers.Select((v, i) => i == 0 ? v.ToString() : i == _layers.Length - 1 ? $"{v} Softmax" : $"{v} ReLU"));

    private void Initialize()
    {
        for (int layer = 0; layer < _weights.Length; layer++)
        {
            int input = _layers[layer];
            int output = _layers[layer + 1];
            double scale = Math.Sqrt(2.0 / input);

            _weights[layer] = new double[output][];
            _biases[layer] = new double[output];

            for (int neuron = 0; neuron < output; neuron++)
            {
                _weights[layer][neuron] = new double[input];
                for (int i = 0; i < input; i++)
                    _weights[layer][neuron][i] = NextGaussian() * scale;
            }
        }
    }

    public double TrainEpoch(IReadOnlyList<DigitSample> samples, double learningRate)
    {
        double loss = 0;
        foreach (var sample in samples)
            loss += TrainSample(sample.Pixels, sample.Label, learningRate);
        return loss / Math.Max(1, samples.Count);
    }

    public int Predict(double[] input, out double confidence)
    {
        var output = Forward(input).Last();
        int best = 0;
        for (int i = 1; i < output.Length; i++)
            if (output[i] > output[best]) best = i;
        confidence = output[best];
        return best;
    }

    public double Accuracy(IReadOnlyList<DigitSample> samples, out int correct)
    {
        correct = 0;
        foreach (var sample in samples)
        {
            int predicted = Predict(sample.Pixels, out _);
            if (predicted == sample.Label) correct++;
        }
        return correct * 100.0 / Math.Max(1, samples.Count);
    }

    private double TrainSample(double[] input, int label, double lr)
    {
        var activations = Forward(input);
        var deltas = new double[_weights.Length][];
        var output = activations[^1];
        double loss = -Math.Log(Math.Max(1e-12, output[label]));

        deltas[^1] = new double[output.Length];
        for (int i = 0; i < output.Length; i++)
            deltas[^1][i] = output[i] - (i == label ? 1.0 : 0.0);

        for (int layer = _weights.Length - 2; layer >= 0; layer--)
        {
            deltas[layer] = new double[_layers[layer + 1]];
            for (int i = 0; i < deltas[layer].Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < deltas[layer + 1].Length; j++)
                    sum += _weights[layer + 1][j][i] * deltas[layer + 1][j];

                deltas[layer][i] = activations[layer + 1][i] > 0 ? sum : 0;
            }
        }

        for (int layer = 0; layer < _weights.Length; layer++)
        {
            var previous = activations[layer];
            for (int neuron = 0; neuron < _weights[layer].Length; neuron++)
            {
                double delta = deltas[layer][neuron];
                for (int i = 0; i < _weights[layer][neuron].Length; i++)
                    _weights[layer][neuron][i] -= lr * delta * previous[i];
                _biases[layer][neuron] -= lr * delta;
            }
        }

        return loss;
    }

    private List<double[]> Forward(double[] input)
    {
        var activations = new List<double[]> { input };
        var current = input;

        for (int layer = 0; layer < _weights.Length; layer++)
        {
            var next = new double[_layers[layer + 1]];
            for (int neuron = 0; neuron < next.Length; neuron++)
            {
                double sum = _biases[layer][neuron];
                for (int i = 0; i < current.Length; i++)
                    sum += _weights[layer][neuron][i] * current[i];
                next[neuron] = layer == _weights.Length - 1 ? sum : Math.Max(0, sum);
            }

            if (layer == _weights.Length - 1)
                next = Softmax(next);

            activations.Add(next);
            current = next;
        }

        return activations;
    }

    private static double[] Softmax(double[] logits)
    {
        double max = logits.Max();
        var result = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = Math.Exp(logits[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < result.Length; i++)
            result[i] /= sum;
        return result;
    }

    private double NextGaussian()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
