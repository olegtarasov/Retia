using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using CLAP;
using Ionic.Zip;
using MathNet.Numerics.LinearAlgebra;
using Retia.Gui;
using Retia.Gui.Models;
using Retia.Gui.Windows;
using Retia.Helpers;
using Retia.Integration;
using Retia.Integration.Helpers;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.RandomGenerator;
using Retia.Training.Data;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;
using Retia.Training.Trainers.Sessions;

namespace SimpleExamples
{
    public partial class Examples
    {
        private const string MnistTrainingImages = "train-images-idx3-ubyte";
        private const string MnistTrainingLabels = "train-labels-idx1-ubyte";
        private const string MnistTestImages = "t10k-images-idx3-ubyte";
        private const string MnistTestLabels = "t10k-labels-idx1-ubyte";
        private const string MnistUrl = "https://github.com/total-world-domination/datasets/raw/master/mnist/mnist.zip";

        [Verb]
        public void Mnist([DefaultValue(true)] bool gui)
        {
            const int batchSize = 128;
            const int hSize = 20;

            MklProvider.TryUseMkl(true, ConsoleProgressWriter.Instance);

            string dataDir = Path.Combine(Path.GetTempPath(), "Retia_datasets", "MNIST");
            DownloadDataset(dataDir);

            Console.WriteLine("Loading training set");
            var trainSet = LoadTrainingSet(dataDir);
            trainSet.BatchSize = batchSize;

            var network = new LayeredNet<float>(batchSize, 1,
                new AffineLayer<float>(trainSet.InputSize, hSize, AffineActivation.Sigmoid),
                new LinearLayer<float>(hSize, trainSet.TargetSize),
                new SoftMaxLayer<float>(trainSet.TargetSize));
            var optimizer = new AdamOptimizer<float>();
            network.Optimizer = optimizer;

            var trainer = new OptimizingTrainer<float>(network, optimizer, trainSet,
                new OptimizingTrainerOptions(1)
                {
                    ErrorFilterSize = 100,
                    MaxEpoch = 1,
                    ProgressWriter = ConsoleProgressWriter.Instance,
                    ReportProgress = new EachIteration(100),
                    ReportMesages = true
                }, new OptimizingSession("MNIST"));

            RetiaGui retiaGui;
            if (gui)
            {
                retiaGui = new RetiaGui();
                retiaGui.RunAsync(() => new TrainingWindow(new TypedTrainingModel<float>(trainer)));
            }

            var runner = ConsoleRunner.Create(trainer, network);
            runner.Run();
        }

        private MnistDataSet LoadTrainingSet(string path)
        {
            string images = Path.Combine(path, MnistTrainingImages);
            string labels = Path.Combine(path, MnistTrainingLabels);

            return MnistDataSet.Load(images, labels);
        }

        private MnistDataSet LoadTestSet(string path)
        {
            string images = Path.Combine(path, MnistTestImages);
            string labels = Path.Combine(path, MnistTestLabels);

            return MnistDataSet.Load(images, labels);
        }

        private void DownloadDataset(string path)
        {
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }

            if (File.Exists(Path.Combine(path, MnistTrainingImages))
                && File.Exists(Path.Combine(path, MnistTrainingLabels)))
            {
                Console.WriteLine("Dataset already downloaded.");
                return;
            }

            string downloadPath = Path.Combine(path, "mnist.zip");
            var downloader = new FileDownloader(ConsoleProgressWriter.Instance);
            if (!downloader.DownloadAndExtract(MnistUrl, downloadPath, file =>
            {
                file.ExtractAll(path, ExtractExistingFileAction.OverwriteSilently);
            }))
            {
                throw new InvalidOperationException("Failed to download MNIST");
            }
        }
    }

    public class MnistDataSet : IDataSet<float>
    {
        private readonly List<MnistImage> _images;

        private MnistDataSet(List<MnistImage> images)
        {
            _images = images;
        }

        public static MnistDataSet Load(string imagesPath, string labelsPath)
        {
            var images = new List<float[]>();
            var result = new List<MnistImage>();

            // Load images
            using (var stream = new FileStream(imagesPath, FileMode.Open, FileAccess.Read))
            using (var reader = stream.NonGreedyReader())
            {
                int magic = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                if (magic != 2051)
                {
                    throw new InvalidOperationException("MNIST image magic error");
                }

                int imageCount = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int rows = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int cols = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int len = rows * cols;

                for (int img = 0; img < imageCount; img++)
                {
                    var data = new float[len];
                    for (int pt = 0; pt < len; pt++)
                    {
                        data[pt] = reader.ReadByte();
                    }

                    images.Add(data);
                }
            }

            // Load labels
            using (var stream = new FileStream(labelsPath, FileMode.Open, FileAccess.Read))
            using (var reader = stream.NonGreedyReader())
            {
                int magic = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                if (magic != 2049)
                {
                    throw new InvalidOperationException("MNIST label magic error");
                }

                int labelCount = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                if (labelCount != images.Count)
                {
                    throw new InvalidOperationException("Label count is not equal to image count!");
                }

                for (int i = 0; i < labelCount; i++)
                {
                    result.Add(new MnistImage(images[i], reader.ReadByte()));
                }
            }

            return new MnistDataSet(result);
        }

        public IDataSet<float> Clone()
        {
            return new MnistDataSet(_images);
        }

        public void Save(Stream stream)
        {
            throw new NotSupportedException();
        }

        public event EventHandler DataSetReset;
        public Sample<float> GetNextSample()
        {
            throw new NotSupportedException();
        }

        public TrainingSequence<float> GetNextSamples(int count)
        {
            if (count != 1) throw new ArgumentOutOfRangeException(nameof(count), "MNIST is not sequential, so sequence length should be set to 1.");
            if (BatchSize <= 0) throw new InvalidOperationException("Set batch size!");

            var gen = SafeRandom.Generator;
            var images = new float[BatchSize][];
            var labels = new float[BatchSize][];
            for (int i = 0; i < BatchSize; i++)
            {
                int idx = gen.Next(_images.Count);
                images[i] = _images[idx].Data;
                labels[i] = new float[10];
                labels[i][_images[idx].Label] = 1.0f;
            }

            return new TrainingSequence<float>(
                new List<Matrix<float>> {Matrix<float>.Build.DenseOfColumnArrays(images)},
                new List<Matrix<float>> { Matrix<float>.Build.DenseOfColumnArrays(labels)});
        }

        public void Reset()
        {
        }

        public int SampleCount => _images.Count;
        public int InputSize => _images[0].Data.Length;
        public int TargetSize => 10;
        public int BatchSize { get; set; }
    }

    public class MnistImage
    {
        public readonly float[] Data;
        public readonly byte Label;

        public MnistImage(float[] data, byte label)
        {
            Data = data;
            Label = label;
        }
    }
}