using System;
using System.IO;
using System.Linq;
using CLAP;
using Retia.Gui;
using Retia.Gui.Models;
using Retia.Gui.Windows;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.ErrorFunctions;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.RandomGenerator;
using Retia.Training.Data;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;

namespace SimpleExamples
{
    public partial class Examples
    {
        private class XorSet : IDataSet<float>
        {
            private int cnt = 0;
            private bool _rand;

            public XorSet(bool random)
            {
                _rand = random;
            }
            public IDataSet<float> Clone()
            {
                throw new NotSupportedException();
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
                var tuples = Enumerable.Range(0, count)
                                       .Select(x =>
                                       {
                                           int a, b;
                                           if (_rand)
                                           {
                                               a = SafeRandom.Generator.Next(2);
                                               b = SafeRandom.Generator.Next(2);
                                           }
                                           else
                                           {
                                               a = cnt & 0x01;
                                               b = (cnt & 0x02) >> 1;
                                           }
                                           cnt++;
                                           return new Tuple<int, int, int>(a, b, a ^ b);
                                       }).ToList();
                return new TrainingSequence<float>(tuples.Select(x => MatrixFactory.Create<float>(2, 1, x.Item1, x.Item2)).ToList(), tuples.Select(x => MatrixFactory.Create<float>(1, 1, x.Item3)).ToList());
            }

            public void Reset()
            {
            }

            public int SampleCount { get; } = 0;
            public int InputSize { get; } = 2;
            public int TargetSize { get; } = 1;
            public int BatchSize { get; } = 1;
        }

        [Verb]
        public void Xor()
        {
            MklProvider.TryUseMkl(true, ConsoleProgressWriter.Instance);

            var optimizer = new RMSPropOptimizer<float>(1e-3f);
            var net = new LayeredNet<float>(1, 1, new AffineLayer<float>(2, 3, AffineActivation.Tanh), new AffineLayer<float>(3, 1, AffineActivation.Tanh) { ErrorFunction = new MeanSquareError<float>() })
            {
                Optimizer = optimizer
            };

            var trainer = new OptimizingTrainer<float>(net, optimizer, new XorSet(true), new OptimizingTrainerOptions
            {
                ErrorFilterSize = 0,
                SequenceLength = 1,
                ReportProgress = new EachIteration(1),
                ReportMesages = true,
                ProgressWriter = ConsoleProgressWriter.Instance,
                LearningRateScaler = new ProportionalLearningRateScaler(new EachIteration(1), optimizer, 9e-5f)
            });

            var runner = ConsoleRunner.Create(trainer, net);

            trainer.TrainReport += (sender, args) =>
            {
                if (args.Errors.Last() < 1e-7f)
                {
                    runner.Stop();
                    Console.WriteLine("Finished training.");
                }
            };

            var gui = new RetiaGui();
            gui.RunAsync(() => new TrainingWindow(new TypedTrainingModel<float>(trainer)));

            runner.Run();
        }

    }
}