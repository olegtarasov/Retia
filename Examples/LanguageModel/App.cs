using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using CLAP;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Single;
using OxyPlot;
using OxyPlot.Wpf;
using Retia.Gui;
using Retia.Gui.Models;
using Retia.Gui.Windows;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.RandomGenerator;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;
using Retia.Training.Trainers.Sessions;
using Window = System.Windows.Window;

// Learn -b="D:\__RNN\ook.bin"
// Prepare -i="D:\__RNN\ook.txt"

namespace LanguageModel
{
	public class App
	{
		private const int BATCH_SIZE = 64;
		private const int SEQ_LEN = 128;
		
		private readonly TextDataProvider _dataProvider = new TextDataProvider(BATCH_SIZE);

	    [Verb]
		public void Compose(
			[Aliases("b"), Required]			string batchesPath,
			[Aliases("n"), Required]			string networkPath,
			[Aliases("o"), Required]			string outPath,
			[Aliases("c"), DefaultValue(10000)] int chars,
            [Aliases("s"), DefaultValue("")]    string startStr)
		{
			_dataProvider.LoadVocab(batchesPath);
			var network = CreateNetwork(networkPath);
		    var vocabLen = _dataProvider.Vocab.Count;
            File.WriteAllText(outPath, TestRNN(network, chars, _dataProvider.Vocab, startStr));
		}


        [Verb]
		public void Prepare(
            [Aliases("i"), Required] string inputFile)
		{
            string batchesPath = Path.Combine(Path.GetDirectoryName(inputFile), Path.GetFileNameWithoutExtension(inputFile) + ".bin");

			Console.WriteLine("Preparing batches...");
			_dataProvider.PrepareTrainingSets(inputFile, ConsoleProgressWriter.Instance);

			Console.WriteLine("Saving batches...");
			_dataProvider.Save(batchesPath);
		}

		[Verb(IsDefault = true)]
		public void Learn(
			[Aliases("b"), Required]	string batchesPath, 
			[Aliases("c")]				string configPath,
			[Aliases("r"), DefaultValue(0.0002f)]  float learningRate,
            [DefaultValue(false)]       bool gpu,
            [DefaultValue(true)]       bool gui)
		{
            MklProvider.TryUseMkl(true, ConsoleProgressWriter.Instance);

            Console.WriteLine($"Loading test set from {batchesPath}");
			_dataProvider.Load(batchesPath);

		    var optimizer = new RMSPropOptimizer<float>(learningRate, 0.95f, 0.0f, 0.9f);
			LayeredNet<float> network;
			if (string.IsNullOrEmpty(configPath))
			{
				network = CreateNetwork(_dataProvider.TrainingSet.InputSize, 128, _dataProvider.TrainingSet.TargetSize);
                network.Optimizer = optimizer;
			}
			else
			{
				network = CreateNetwork(configPath);
				network.Optimizer = optimizer;
			}
            network.ResetOptimizer();

		    if (gpu)
		    {
		        Console.WriteLine("Brace yourself for GPU!");
		        network.UseGpu();
		    }

			var trainOptions = new OptimizingTrainerOptions(SEQ_LEN)
		                       {
		                           ErrorFilterSize = 100,
		                           ReportMesages = true,
		                           MaxEpoch = 1000,
                                   ProgressWriter = ConsoleProgressWriter.Instance,
                                   ReportProgress = new EachIteration(10)
		                       };

            trainOptions.LearningRateScaler = new ProportionalLearningRateScaler(new ActionSchedule(1, PeriodType.Iteration), 9.9e-5f);

            var session = new OptimizingSession(Path.GetFileNameWithoutExtension(batchesPath));
            var trainer = new OptimizingTrainer<float>(network, optimizer, _dataProvider.TrainingSet, trainOptions, session);

            RetiaGui retiaGui;
            TypedTrainingModel<float> model = null;
            if (gui)
            {
                retiaGui = new RetiaGui();
                retiaGui.RunAsync(() =>
                {
                    model = new TypedTrainingModel<float>(trainer);
                    return new TrainingWindow(model);
                });
            }

            var epochWatch = new Stopwatch();
           
			trainer.EpochReached += sess =>
		    {
                epochWatch.Stop();
		        Console.WriteLine($"Trained epoch in {epochWatch.Elapsed.TotalSeconds} s.");

                // Showcasing plot export
                if (model != null)
                {
                    using (var stream = new MemoryStream())
                    {
                        model.ExportErrorPlot(stream, 600, 400);

                        stream.Seek(0, SeekOrigin.Begin);

                        session.AddFileToReport("ErrorPlots\\plot.png", stream);
                    }
                }

                epochWatch.Restart();
		    };
            trainer.PeriodicActions.Add(new UserAction(new ActionSchedule(100, PeriodType.Iteration), () =>
            {
                if (gpu)
                {
                    network.TransferStateToHost();
                }

                string text = TestRNN(network.Clone(1, SEQ_LEN), 500, _dataProvider.Vocab);
                Console.WriteLine(text);
                session.AddFileToReport("Generated\\text.txt", text);

                trainOptions.ProgressWriter.ItemComplete();
            }));

            var runner = ConsoleRunner.Create(trainer, network);
            epochWatch.Start();
            runner.Run();
		}
  
        private static LayeredNet<float> CreateNetwork(string fileName)
		{
			return LayeredNet<float>.Load(fileName);
		}

		private static LayeredNet<float> CreateNetwork(int xSize, int hSize, int ySize)
		{
            return new LayeredNet<float>(BATCH_SIZE, SEQ_LEN, 
                new GruLayer<float>(xSize, hSize),
                new LinearLayer<float>(hSize, ySize),
                new SoftMaxLayer<float>(ySize));
        }

		private static string TestRNN(NeuralNet<float> network, int count, List<char> vocab, string startFrom = "")
		{
			var rnd = SafeRandom.Generator;
			
			if (startFrom == "")
			{
                char start;
				do
				{
					start = char.ToUpper(vocab[rnd.Next(vocab.Count)]);
				} while (!vocab.Contains(start));
                startFrom = new string(start, 1);
			}


			var input = new DenseMatrix(vocab.Count, 1);

			network.ResetMemory();
			var sb = new StringBuilder();
		    sb.Append(startFrom);

            for (int i = 0; i < startFrom.Length - 1; i++)
		    {
		        input[vocab.IndexOf(startFrom[i]), 0] = 1;
                network.Step(input);
                input[vocab.IndexOf(startFrom[i]), 0] = 0;
            }

            input[vocab.IndexOf(startFrom[startFrom.Length-1]), 0] = 1;

			for (int i = 0; i < count; i++)
			{
				var p = network.Step(input);
				var index = MathProvider<float>.Instance.SoftMaxChoice(p)[0];
				char output = vocab[index];

				sb.Append(output);
			    input = new DenseMatrix(vocab.Count, 1)
			            {
			                [vocab.IndexOf(output), 0] = 1
			            };
			}

			return sb.ToString();
		}  
    }
}