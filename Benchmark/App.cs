using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using CLAP;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.Common.Mkl;
using Retia.Gui;
using Retia.Gui.Models;
using Retia.Gui.Windows;
using Retia.Integration;
using Retia.Mathematics;
#if !CPUONLY
using Retia.NativeWrapper;
#endif
using Retia.Neural;
using Retia.Neural.ErrorFunctions;
using Retia.Neural.Initializers;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.RandomGenerator;
using Retia.Training.Data;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;
using df = System.Double;

namespace Benchmark
{
    public class App
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

        [Verb]
        public void TestXor()
        {
            //var optimizer = new RMSPropOptimizer<float>(1e-7f, 0.0f, 0.0f, 0.9f);
            var optimizer = new SGDOptimizer<float>(0.1f);
            var net = new LayeredNet<float>(1, 1, new LinearLayer<float>(2, 2), new TanhLayer<float>(2), new LinearLayer<float>(2, 1), new SigmoidLayer<float>(1) {ErrorFunction = new CrossEntropyError<float>()})
            {
                Optimizer = optimizer
            };

            //double err = double.MaxValue;
            //while (err > 1e-10f)
            //{
            //    int a = SafeRandom.Generator.Next(2), b = SafeRandom.Generator.Next(2);
            //    int c = a ^ b;

            //    var input = MatrixFactory.Create<float>(2, 1, a, b);
            //    var target = MatrixFactory.Create<float>(1, 1, c);

            //    net.InitSequence();
            //    var output = net.Step(input, true);

            //    err = net.Error(output, target);

            //    Console.WriteLine($"Err: {err:0.00000000}");

            //    net.BackPropagate(new List<Matrix<float>> {target});
            //    net.Optimize();

            //    Thread.Sleep(100);
            //}

            var trainer = new OptimizingTrainer<float>(net, optimizer, null, new OptimizingTrainerOptions
            {
                ErrorFilterSize = 0,
                SequenceLength = 1,
                ReportProgress = new EachIteration(1),
                ReportMesages = true,
                ProgressWriter = ConsoleProgressWriter.Instance,
                LearningRateScaler = new ProportionalLearningRateScaler(new EachIteration(1), optimizer, 9e-5f)
            })
            {
                TrainingSet = new XorSet()
            };

            trainer.TrainReport += (sender, args) =>
            {
                var n = net;
                if (args.Errors.Last() < 1e-7f)
                {
                }

                //Thread.Sleep(100);
            };

            var gui = new RetiaGui();
            gui.RunAsync(() => new TrainingWindow(new TypedTrainingModel<float>(trainer)));

            ConsoleRunner.RunTrainer(trainer, net);
        }

#if !CPUONLY
        [Verb]
        public void TestGpuLayers()
        {
            var dataSet = new TestDataSet<float>(3, 4, 2, 5);

            Console.WriteLine("Testing softmax forward");
            var softmaxLayer = new SoftMaxLayer<float>(dataSet.InputSize);
            TestLayerForward(softmaxLayer, dataSet, dataSet.InputSize);

            Console.WriteLine("Testing linear forward");
            var linLayer = new LinearLayer<float>(dataSet.InputSize, dataSet.TargetSize, new RandomMatrixInitializer<float>());
            TestLayerForward(linLayer, dataSet);

            Console.WriteLine("Testing GRU forward");
            var gruLayer = new GruLayer<float>(dataSet.InputSize, dataSet.TargetSize, new ProportionalRandomMatrixInitializer<float>(), new ProportionalRandomMatrixInitializer<float>(), new RandomMatrixInitializer<float>());
            TestLayerForward(gruLayer, dataSet);
        }

        private void TestLayerForward(NeuroLayer<float> layer, TestDataSet<float> dataSet, int? outSize = null)
        {
            layer.Initialize(dataSet.BatchSize, dataSet.SampleCount);

            int finalOutSize = outSize.GetValueOrDefault(dataSet.TargetSize);
            var gpuTester = new LayerTester(layer.CreateSpec());

            for (int i = 0; i < 3; i++)
            {
                var seq = dataSet.GetNextSamples(dataSet.SampleCount);
                var cpuOut = new List<Matrix<float>>();
                foreach (var input in seq.Inputs)
                {
                    cpuOut.Add(layer.Step(input));
                }

                var gpuOut = Enumerable.Range(0, dataSet.SampleCount).Select(x => Matrix<float>.Build.Dense(finalOutSize, dataSet.BatchSize)).ToList();
                gpuTester.TestForward(seq.Inputs, gpuOut);

                //Console.WriteLine("CPU output:");
                //foreach (var matrix in cpuOut)
                //{
                //    Console.WriteLine(matrix.ToMatrixString());
                //    Console.WriteLine("---------------");
                //}

                //Console.WriteLine("GPU output:");
                //foreach (var matrix in gpuOut)
                //{
                //    Console.WriteLine(matrix.ToMatrixString());
                //    Console.WriteLine("---------------");
                //}

                for (int j = 0; j < dataSet.SampleCount; j++)
                {
                    var cpuArr = cpuOut[j].AsColumnMajorArray();
                    var gpuArr = gpuOut[j].AsColumnMajorArray();

                    for (int k = 0; k < cpuArr.Length; k++)
                    {
                        if (Math.Abs(cpuArr[k] - gpuArr[k]) > 1e-4f)
                        {
                            Console.WriteLine($"FAILED on iteration {i}");
                        }
                    }
                }

                layer.ResetMemory();
            }

            gpuTester.Dispose();
        }
#endif
    }
}