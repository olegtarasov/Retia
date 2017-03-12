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

        private void TestLayerForward(LayerBase<float> layer, TestDataSet<float> dataSet, int? outSize = null)
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