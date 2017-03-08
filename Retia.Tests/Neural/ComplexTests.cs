using System;
using System.IO;
using System.Linq;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.RandomGenerator;
using Retia.Training.Data;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;
using Xunit;

namespace Retia.Tests.Neural
{
    public class ComplexTests
    {
        private class XorSet : IDataSet<float>
        {
            public IDataSet<float> Clone()
            {
                throw new NotImplementedException();
            }

            public void Save(Stream stream)
            {
                throw new NotImplementedException();
            }

            public event EventHandler DataSetReset;
            public Sample<float> GetNextSample()
            {
                throw new NotImplementedException();
            }

            public TrainingSequence<float> GetNextSamples(int count)
            {
                var tuples = Enumerable.Range(0, count)
                                       .Select(x =>
                                       {
                                           int a = SafeRandom.Generator.Next(2);
                                           int b = SafeRandom.Generator.Next(2);

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

        //[Fact]
        public void CanLearnXor()
        {
            var optimizer = new RMSPropOptimizer<float>(2e-7f, 0.0f, 0.0f, 0.0f);
            var net = new LayeredNet<float>(1, 1, new LinearLayer<float>(2, 1), new LinearLayer<float>(1, 1))
                      {
                          Optimizer = optimizer
                      };

            var trainer = new OptimizingTrainer<float>(net, optimizer, null, new OptimizingTrainerOptions
                                                                             {
                                                                                 ErrorFilterSize = 0,
                                                                                 SequenceLength = 1
                                                                                 //ReportProgress = new EachIteration(1)
                                                                             })
                          {
                              TrainingSet = new XorSet()
                          };


            trainer.SequenceTrained += () =>
            {
                var n = net;
            };
            trainer.Train(CancellationToken.None).Wait();
        }
    }
}