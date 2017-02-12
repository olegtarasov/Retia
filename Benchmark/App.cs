using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CLAP;
using Retia.Neural;
using Retia.Neural.Layers;
using Retia.Optimizers;

namespace Benchmark
{
    public class App
    {
        //[Verb]
        //public void TestGpuLayers()
        //{
        //    var dataSet = new TestDataSet(3, 4, 2, 5);

        //    Console.WriteLine("Testing softmax forward");
        //    var softmaxLayer = new SoftMaxLayer(dataSet.InputSize);
        //    TestLayerForward(softmaxLayer, dataSet, dataSet.InputSize);

        //    Console.WriteLine("Testing linear forward");
        //    var linLayer = new LinearLayer(dataSet.InputSize, dataSet.TargetSize);
        //    TestLayerForward(linLayer, dataSet);

        //    Console.WriteLine("Testing GRU forward");
        //    var gruLayer = new GruLayer(dataSet.InputSize, dataSet.TargetSize);
        //    TestLayerForward(gruLayer, dataSet);
        //}

        [Verb]
        public void CheckGrad()
        {
            const int seqLen = 5;

            //var net = new LayeredNet(1, seqLen, new GruLayer(6, 3), new LinearLayer(3, 2), new SoftMaxLayer(2))
            //          {
            //              Optimizer = new RMSPropOptimizer()
            //          };

            //LayeredNet.CheckGrad(net, seqLen);
        }

        //private void TestLayerForward(NeuroLayer layer, TestDataSet dataSet, int? outSize = null)
        //{
        //    layer.Initialize(dataSet.BatchSize, dataSet.SampleCount);

        //    int finalOutSize = outSize.GetValueOrDefault(dataSet.TargetSize);
        //    var gpuTester = new LayerTester(layer.CreateSpec());

        //    for (int i = 0; i < 3; i++)
        //    {
        //        var seq = dataSet.GetNextSamples(dataSet.SampleCount);
        //        var cpuOut = new List<Matrix>();
        //        foreach (var input in seq.Inputs)
        //        {
        //            cpuOut.Add(layer.Step(input));
        //        }

        //        var gpuOut = Enumerable.Range(0, dataSet.SampleCount).Select(x => (Matrix)new DenseMatrix(finalOutSize, dataSet.BatchSize)).ToList();
        //        gpuTester.TestForward(seq.Inputs, gpuOut);

        //        //Console.WriteLine("CPU output:");
        //        //foreach (var matrix in cpuOut)
        //        //{
        //        //    Console.WriteLine(matrix.ToMatrixString());
        //        //    Console.WriteLine("---------------");
        //        //}

        //        //Console.WriteLine("GPU output:");
        //        //foreach (var matrix in gpuOut)
        //        //{
        //        //    Console.WriteLine(matrix.ToMatrixString());
        //        //    Console.WriteLine("---------------");
        //        //}

        //        for (int j = 0; j < dataSet.SampleCount; j++)
        //        {
        //            var cpuArr = cpuOut[j].AsColumnMajorArray();
        //            var gpuArr = gpuOut[j].AsColumnMajorArray();

        //            for (int k = 0; k < cpuArr.Length; k++)
        //            {
        //                if (Math.Abs(cpuArr[k] - gpuArr[k]) > 1e-4f)
        //                {
        //                    Console.WriteLine($"FAILED on iteration {i}");
        //                }
        //            }
        //        }

        //        layer.ResetMemory();
        //    }

        //    gpuTester.Dispose();
        //}
    }
}