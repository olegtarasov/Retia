#if !CPUONLY
using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Interop;
using Retia.Neural.Layers;
using Retia.Tests.Neural;
using Retia.Tests.Plumbing;
using Xunit;

namespace Retia.Tests.Gpu
{
    public class GpuLayersTests
    {
        [Fact]
        public void CanComputeForwardLinear()
        {
            var dataSet = new TestDataSet<float>(3, 4, 5, 10);

            var linLayer = new LinearLayer<float>(dataSet.InputSize, dataSet.TargetSize);
            TestGpuLayer(linLayer, dataSet);
        }

        [Fact]
        public void CanComputeSoftmaxForward()
        {
            var dataSet = new TestDataSet<float>(3, 4, 5, 10);

            var softmaxLayer = new SoftMaxLayer<float>(dataSet.InputSize);
            TestGpuLayer(softmaxLayer, dataSet, dataSet.InputSize);
        }

        [Fact]
        public void CanComputeGruForward()
        {
            var dataSet = new TestDataSet<float>(3, 4, 5, 10);

            var gruLayer = new GruLayer<float>(dataSet.InputSize, dataSet.TargetSize);
            TestGpuLayer(gruLayer, dataSet);
        }

        [Fact]
        public void CanComputeMultiGruForward()
        {
            var dataSet = new TestDataSet<float>(3, 4, 5, 10);

            for (int cnt = 1; cnt < 5; cnt++)
            {
                var gruLayer = new MultiGruLayer<float>(dataSet.InputSize, dataSet.TargetSize, cnt);
                TestGpuLayer(gruLayer, dataSet);
            }
        }

        private unsafe void TestGpuLayer(LayerBase<float> layer, TestDataSet<float> dataSet, int? outSize = null)
        {
            layer.Initialize(dataSet.BatchSize, dataSet.SampleCount);
            var gpuLayer = layer.CreateGpuLayer();

            int finalOutSize = outSize.GetValueOrDefault(dataSet.TargetSize);

            for (int step = 0; step < 3; step++)
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

                for (int sample = 0; sample < dataSet.SampleCount; sample++)
                {
                    cpuOut[sample].ShouldMatrixEqualWithinError(gpuOut[sample]);
                }
            }
        }
    }
}

#endif