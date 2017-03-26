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
using Retia.Interop;
using Retia.Gui;
using Retia.Gui.Models;
using Retia.Gui.Windows;
using Retia.Integration;
using Retia.Mathematics;
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
            var dataSet = new TestDataSet<float>(3, 4, 5, 10);

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

        private unsafe void TestLayerForward(LayerBase<float> layer, TestDataSet<float> dataSet, int? outSize = null)
        {
            layer.Initialize(dataSet.BatchSize, dataSet.SampleCount);
            var gpuLayer = layer.CreateGpuLayer();

            int finalOutSize = outSize.GetValueOrDefault(dataSet.TargetSize);
            
            for (int i = 0; i < 3; i++)
            {
                var seq = dataSet.GetNextSamples(dataSet.SampleCount);
                var cpuOut = new List<Matrix<float>>();
                layer.InitSequence();
                foreach (var input in seq.Inputs)
                {
                    cpuOut.Add(layer.Step(input));
                }

                var gpuOut = Enumerable.Range(0, dataSet.SampleCount).Select(x => Matrix<float>.Build.Dense(finalOutSize, dataSet.BatchSize)).ToArray();
                using (var inPtrs = new MatrixPointersBag<float>(true, seq.Inputs.ToArray()))
                using (var outPtrs = new MatrixPointersBag<float>(true, gpuOut))
                {
                    fixed (MatrixDefinition* inDef = &inPtrs.Definitions[0], outDef = &outPtrs.Definitions[0])
                    {
                        GpuInterface.LayerForwardSequence(gpuLayer, inDef, seq.Inputs.Count);
                        GpuInterface.TransferLayerOutputsToHost(gpuLayer, outDef, gpuOut.Length);
                    }
                }

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

                //Console.WriteLine("================================");

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
            }
        }
#endif
    }
}