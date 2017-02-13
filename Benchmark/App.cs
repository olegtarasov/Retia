using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CLAP;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.Common.Mkl;
using Retia.Neural;
using Retia.Neural.Initializers;
using Retia.Neural.Layers;
using Retia.Optimizers;

using df = System.Double;

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
            Control.UseNativeMKL();

            //const int seqLen = 5;

            //var net = new LayeredNet(1, seqLen, new GruLayer(6, 3), new LinearLayer(3, 2), new SoftMaxLayer(2))
            //          {
            //              Optimizer = new RMSPropOptimizer()
            //          };

            //LayeredNet.CheckGrad(net, seqLen);

            const df delta = 1e-5f;

            var dataSet = new TestDataSet<df>(3, 2, 5, 10);
            var layer = new LinearLayer<df>(dataSet.InputSize, dataSet.TargetSize);
            layer.Initialize(dataSet.BatchSize, dataSet.SampleCount);
            layer.InitSequence();

            var samples = dataSet.GetNextSamples(dataSet.SampleCount);

            var outputs = new List<Matrix<df>>();
            for (int i = 0; i < samples.Inputs.Count; i++)
            {
                outputs.Add(layer.Step(samples.Inputs[i], true));
            }

            layer.BackPropagate(ErrorPropagate(outputs, samples.Targets));

            for (int i = 0; i < layer.TotalParamCount; i++)
            {
                var pLayer = layer.Clone();
                var nLayer = layer.Clone();

                pLayer.InitSequence();
                nLayer.InitSequence();

                pLayer.SetParam(i, pLayer.GetParam(i) + delta);
                nLayer.SetParam(i, nLayer.GetParam(i) - delta);

                double pErr = 0.0, nErr = 0.0;
                for (int j = 0; j < samples.Inputs.Count; j++)
                {
                    pErr += pLayer.LayerError(pLayer.Step(samples.Inputs[j]), samples.Targets[j]);
                    nErr += nLayer.LayerError(nLayer.Step(samples.Inputs[j]), samples.Targets[j]);
                }

                double num = (pErr - nErr) / (2.0f * delta);
                double real = layer.GetParam(i, true);
                double d = num - real;

                if (Math.Abs(d) > 1e-7)
                {
                    Console.WriteLine("Fuck");
                }
            }
        }

        private List<Matrix<df>> ErrorPropagate(List<Matrix<df>> outputs, List<Matrix<df>> targets)
        {
            var result = new List<Matrix<df>>();

            for (int i = 0; i < outputs.Count; i++)
            {
                var r = Matrix<df>.Build.Dense(outputs[0].RowCount, outputs[0].ColumnCount);
                var oa = outputs[i].AsColumnMajorArray();
                var ta = targets[i].AsColumnMajorArray();
                var ra = r.AsColumnMajorArray();

                for (int j = 0; j < oa.Length; j++)
                {
                    ra[j] = (oa[j] - ta[j]) / (df)oa.Length;
                }

                result.Add(r);
            }

            return result;
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