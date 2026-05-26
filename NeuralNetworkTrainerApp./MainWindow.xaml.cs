using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using NeuralNetworkLab.Core;
using Windows.Graphics;

namespace NeuralNetworkTrainerApp;

public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        SetWindowSize(1280, 820);
        ResetResultState();
    }

    private void SetWindowSize(int width, int height)
    {
        if (AppWindow is null)
        {
            return;
        }

        AppWindow.Resize(new SizeInt32(width, height));
        AppWindow.Title = "Neural Network Trainer";

        var iconPath = System.IO.Path.Combine(AppContext.BaseDirectory, "Assets", "AppIcon.ico");
        if (File.Exists(iconPath))
        {
            AppWindow.SetIcon(iconPath);
        }
    }

    private void ContentGrid_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        bool narrow = e.NewSize.Width < 920;
        bool compactWorkspace = e.NewSize.Width < 1080;
        bool compactMetrics = e.NewSize.Width < 760;

        SettingsColumn.Width = narrow ? new GridLength(1, GridUnitType.Star) : new GridLength(360);
        WorkspaceColumn.Width = narrow ? new GridLength(0) : new GridLength(1, GridUnitType.Star);

        Grid.SetColumn(SettingsPanel, 0);
        Grid.SetRow(SettingsPanel, 0);
        Grid.SetColumn(WorkspacePanel, narrow ? 0 : 1);
        Grid.SetRow(WorkspacePanel, narrow ? 1 : 0);

        ApplyMetricsLayout(compactMetrics);
        ApplyDetailsLayout(compactWorkspace);
        ApplyDiagnosticsLayout(compactWorkspace);
    }

    private void ApplyMetricsLayout(bool compact)
    {
        MetricColumnOne.Width = new GridLength(1, GridUnitType.Star);
        MetricColumnTwo.Width = compact ? new GridLength(0) : new GridLength(1, GridUnitType.Star);
        MetricColumnThree.Width = compact ? new GridLength(0) : new GridLength(1, GridUnitType.Star);

        Grid.SetColumn(AccuracyCard, 0);
        Grid.SetRow(AccuracyCard, 0);
        Grid.SetColumn(CorrectCard, compact ? 0 : 1);
        Grid.SetRow(CorrectCard, compact ? 1 : 0);
        Grid.SetColumn(ArchitectureCard, compact ? 0 : 2);
        Grid.SetRow(ArchitectureCard, compact ? 2 : 0);
    }

    private void ApplyDetailsLayout(bool compact)
    {
        HistoryColumn.Width = new GridLength(1, GridUnitType.Star);
        PredictionsColumn.Width = compact ? new GridLength(0) : new GridLength(1.1, GridUnitType.Star);

        Grid.SetColumn(PredictionsCard, compact ? 0 : 1);
        Grid.SetRow(PredictionsCard, compact ? 1 : 0);
    }

    private void ApplyDiagnosticsLayout(bool compact)
    {
        MistakeColumn.Width = compact ? new GridLength(1, GridUnitType.Star) : new GridLength(320);
        LogColumn.Width = compact ? new GridLength(0) : new GridLength(1, GridUnitType.Star);

        Grid.SetColumn(LogCard, compact ? 0 : 1);
        Grid.SetRow(LogCard, compact ? 1 : 0);
    }

    private void SmallPreset_Click(object sender, RoutedEventArgs e)
    {
        SetPreset(5000, 2000, 128, 5, false);
    }

    private void LargePreset_Click(object sender, RoutedEventArgs e)
    {
        SetPreset(100000, 5000, 128, 10, false);
    }

    private void SetPreset(int train, int test, int hidden, int epochs, bool secondLayer)
    {
        TrainSizeBox.Value = train;
        TestSizeBox.Value = test;
        HiddenBox.Value = hidden;
        EpochsBox.Value = epochs;
        SecondLayerSwitch.IsOn = secondLayer;
    }

    private async void TrainButton_Click(object sender, RoutedEventArgs e)
    {
        TrainButton.IsEnabled = false;
        TrainingProgress.IsActive = true;
        StatusText.Text = "Навчання";
        ResetResultState();
        AppendLog("Запуск навчання...");

        var config = new TrainingConfiguration
        {
            TrainSize = ReadNumber(TrainSizeBox, 5000, 100, 100000),
            TestSize = ReadNumber(TestSizeBox, 2000, 100, 10000),
            HiddenNeurons = ReadNumber(HiddenBox, 128, 16, 512),
            Epochs = ReadNumber(EpochsBox, 5, 1, 50),
            UseSecondHiddenLayer = SecondLayerSwitch.IsOn
        };

        var progress = new Progress<string>(AppendLog);

        try
        {
            var result = await Task.Run(() => new NeuralNetworkTrainer().Run(config, progress));
            RenderResult(result);
            StatusText.Text = "Завершено";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Помилка";
            AppendLog($"Помилка: {ex.Message}");
        }
        finally
        {
            TrainingProgress.IsActive = false;
            TrainButton.IsEnabled = true;
        }
    }

    private static int ReadNumber(NumberBox box, int fallback, int min, int max)
    {
        if (double.IsNaN(box.Value))
        {
            box.Value = fallback;
            return fallback;
        }

        int value = (int)Math.Round(box.Value);
        value = Math.Clamp(value, min, max);
        box.Value = value;
        return value;
    }

    private void ResetResultState()
    {
        AccuracyText.Text = "--";
        CorrectText.Text = "--";
        ArchitectureText.Text = "784 -> 128 -> 10";
        AccuracyBar.Value = 0;
        HistoryList.ItemsSource = Array.Empty<EpochRowView>();
        PredictionsList.ItemsSource = Array.Empty<PredictionRowView>();
        MistakeText.Text = "З'явиться після запуску.";
        MistakeCanvas.Children.Clear();
        LogBox.Text = string.Empty;
    }

    private void RenderResult(TrainingResult result)
    {
        AccuracyText.Text = $"{result.Accuracy:F2}%";
        CorrectText.Text = $"{result.Correct} / {result.Total}";
        ArchitectureText.Text = result.Architecture;
        AccuracyBar.Value = result.Accuracy;

        HistoryList.ItemsSource = result.History
            .Select(metric => new EpochRowView(
                $"#{metric.Epoch}",
                $"Train {metric.TrainAccuracy:F2}% / Test {metric.TestAccuracy:F2}%",
                $"Loss {metric.Loss:F4}",
                $"{metric.TestAccuracy:F1}%"))
            .ToList();

        PredictionsList.ItemsSource = result.FirstPredictions
            .Select(row => new PredictionRowView(
                row.Index.ToString(),
                $"Очікувано {row.Expected}, прогноз {row.Predicted}",
                $"Впевненість {row.Confidence * 100:F1}%",
                row.Expected == row.Predicted ? "OK" : "Помилка",
                row.Expected == row.Predicted ? SuccessBrush : WarningBrush))
            .ToList();

        if (result.FirstMistake is null)
        {
            MistakeText.Text = "На тестовій вибірці помилок не знайдено.";
            AppendLog("На тестовій вибірці помилок не знайдено.");
            return;
        }

        MistakeText.Text = $"Очікувано {result.FirstMistake.Label}, прогноз {result.FirstMistakePredicted}, впевненість {result.FirstMistakeConfidence * 100:F1}%";
        AppendLog($"Перша помилка: очікувана цифра {result.FirstMistake.Label}, прогноз {result.FirstMistakePredicted}, впевненість {result.FirstMistakeConfidence * 100:F1}%");
        RenderDigit(result.FirstMistake.Pixels);
    }

    private void AppendLog(string line)
    {
        LogBox.Text += $"{line}{Environment.NewLine}";
        LogBox.SelectionStart = LogBox.Text.Length;
    }

    private void RenderDigit(double[] pixels)
    {
        const int scale = 8;

        MistakeCanvas.Children.Clear();

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                byte value = (byte)(255 - (int)Math.Round(pixels[y * 28 + x] * 255));
                var rectangle = new Rectangle
                {
                    Width = scale,
                    Height = scale,
                    Fill = new SolidColorBrush(Windows.UI.Color.FromArgb(255, value, value, value))
                };

                Canvas.SetLeft(rectangle, x * scale);
                Canvas.SetTop(rectangle, y * scale);
                MistakeCanvas.Children.Add(rectangle);
            }
        }
    }

    private static SolidColorBrush SuccessBrush { get; } = new(Windows.UI.Color.FromArgb(255, 5, 150, 105));
    private static SolidColorBrush WarningBrush { get; } = new(Windows.UI.Color.FromArgb(255, 217, 119, 6));
}

public sealed record EpochRowView(string EpochText, string AccuracyText, string LossText, string TestText);

public sealed record PredictionRowView(
    string IndexText,
    string PredictionText,
    string ConfidenceText,
    string ResultText,
    Brush ResultBrush);
