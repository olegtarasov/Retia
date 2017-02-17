using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using CLAP;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.RandomGenerator;
using Retia.Training.Trainers;

// Learn -b="D:\__RNN\ook.bin"
// Prepare -i="D:\__RNN\ook.txt"

namespace LanguageModel
{
	public class App
	{
		private const int BATCH_SIZE = 64;
		private const int SEQ_LEN = 64;
		
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
            [DefaultValue(false)]       bool gpu)
		{
            Control.UseNativeMKL();

		    string rnnPath = $"NETWORK_{DateTime.Now:dd_MM_yy-HH_mm_ss}.rnn";
			var errFile = new FileStream($"ERR_{DateTime.Now:dd_MM_yy-HH_mm_ss}.txt", FileMode.Create);
			var errWriter = new StreamWriter(errFile);

			Console.WriteLine($"Loading test set from {batchesPath}");
			_dataProvider.Load(batchesPath);

		    var optimizer = new RMSPropOptimizer<float>(learningRate, 0.95f, 0.0f, 0.9f);
			LayeredNet<float> network;
			if (string.IsNullOrEmpty(configPath))
			{
				network = CreateNetwork(_dataProvider.InputSize, 128, _dataProvider.OutputSize, 1);
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

			var cts = new CancellationTokenSource();
		    var trainOptions = new OptimizingTrainerOptions
		                       {
		                           ErrorFilterSize = 20,
		                           SequenceLength = SEQ_LEN,
		                           ReportMesages = true,
		                           MaxEpoch = 0
		                       };

            trainOptions.RunTests.Never();
            trainOptions.ScaleLearningRate.EachEpoch(10, 0.97);
            trainOptions.ReportProgress.EachIteration(10);
            trainOptions.RunUserTests.EachIteration(100);
			var trainer = new OptimizingTrainer<float>(network, optimizer, _dataProvider, null, trainOptions);
            var epochWatch = new Stopwatch();
           
			trainer.Message += (sender, args) => Console.WriteLine(args.Message);
			trainer.TrainReport += (sender, args) =>
			{
				foreach (var err in args.Errors)
				{
					errWriter.WriteLine($"{DateTime.Now.TimeOfDay};{err.ToString(NumberFormatInfo.InvariantInfo)}");
				}
                errWriter.Flush();
				errFile.Flush(true);
				network.Save(rnnPath);
			};
			trainer.UserTest += (sender, args) =>
			{
			    if (gpu)
			    {
			        network.TransferStateToHost();
			    }
			    Console.WriteLine(TestRNN(network.Clone(), 500, _dataProvider.Vocab));
			};
		    trainer.EpochReached += () =>
		    {
                epochWatch.Stop();
		        Console.WriteLine($"Trained epoch in {epochWatch.Elapsed.TotalSeconds} s.");
		        //Console.ReadKey();
		    };
		   
			var task = trainer.Train(cts.Token);
            epochWatch.Start();
		    var running = true;
			while (running)
			{
                var c=Console.ReadKey().Key;
			    switch (c)
			    {
                    case ConsoleKey.Q:
			            running = false;
                        break;
                
                    case ConsoleKey.R:
                        Console.WriteLine("Reseting optimizer cache!");
                        trainer.Pause();
                        network.ResetOptimizer();
                        trainer.Resume();
                        break;
			    }
            }

			cts.Cancel();
			Task.WaitAll(task);
		}

  
        private static LayeredNet<float> CreateNetwork(string fileName)
		{
			return LayeredNet<float>.Load(fileName);
		}

		private static LayeredNet<float> CreateNetwork(int xSize, int hSize, int ySize, int layerCount)
		{
            return new LayeredNet<float>(BATCH_SIZE, SEQ_LEN, 
                new GruLayer<float>(xSize, hSize),
                //new GruLayer(hSize, hSize),
                //new GruLayer(hSize, hSize),
                //new GruLayer(hSize, hSize),
                //new GruLayer(hSize, hSize),
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