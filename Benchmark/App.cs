using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CLAP;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Neural.Layers;

namespace Benchmark
{
    public class App
    {
        [Verb]
        public void DotMatrix([DefaultValue(10)] int count, [DefaultValue(4096)] int dimension, [DefaultValue(false)] bool gpu)
        {
            Console.WriteLine("========= Math.NET matrices");
            var dist = new Normal();

            if (gpu)
            {
                Control.UseNativeCUDA();
            }
            else
            {
                Control.UseNativeMKL();
            }

            var mat = Enumerable.Range(0, count * 2).Select(x => DenseMatrix.CreateRandom(dimension, dimension, dist)).ToArray();
            var watch = new Stopwatch();
            watch.Start();

            for (int i = 0; i < count; i++)
            {
                var w = new Stopwatch();
                w.Start();

                var m1 = mat[i * 2];
                var m2 = mat[i * 2 + 1];

                var result = m1 * m2;
                w.Stop();

                Console.WriteLine($"{i + 1} completed in {w.Elapsed.TotalSeconds} s.");
            }
            watch.Stop();

            Console.WriteLine($"Total time: {watch.Elapsed.TotalSeconds} s.");
        }

        [Verb]
        public void TestGpuLayers()
        {
            var dataSet = new TestDataSet(3, 4, 2, 5);

            Console.WriteLine("Testing softmax forward");
            var softmaxLayer = new SoftMaxLayer(dataSet.InputSize);
            TestLayerForward(softmaxLayer, dataSet, dataSet.InputSize);

            Console.WriteLine("Testing linear forward");
            var linLayer = new LinearLayer(dataSet.InputSize, dataSet.TargetSize);
            TestLayerForward(linLayer, dataSet);

            Console.WriteLine("Testing GRU forward");
            var gruLayer = new GruLayer(dataSet.InputSize, dataSet.TargetSize);
            TestLayerForward(gruLayer, dataSet);
        }

        private void TestLayerForward(NeuroLayer layer, TestDataSet dataSet, int? outSize = null)
        {
            layer.Initialize(dataSet.BatchSize, dataSet.SampleCount);

            int finalOutSize = outSize.GetValueOrDefault(dataSet.TargetSize);
            var gpuTester = new LayerTester(layer.CreateSpec());

            for (int i = 0; i < 3; i++)
            {
                var seq = dataSet.GetNextSamples(dataSet.SampleCount);
                var cpuOut = new List<Matrix>();
                foreach (var input in seq.Inputs)
                {
                    cpuOut.Add(layer.Step(input));
                }

                var gpuOut = Enumerable.Range(0, dataSet.SampleCount).Select(x => (Matrix)new DenseMatrix(finalOutSize, dataSet.BatchSize)).ToList();
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
    }
}