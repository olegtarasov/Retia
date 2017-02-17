using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;
using Retia.Neural.Layers;
using Xunit;
using XunitShould;

namespace Retia.Tests.Neural
{
    public class GradientCheckTests
    {
        [Fact]
        public void CanGradientCheckLinearLayer()
        {
            var layer = new LinearLayer<double>(5, 3);
            TestLayer(layer);
        }

        [Fact]
        public void CanGradientCheckGruLayer()
        {
            var layer = new GruLayer<double>(5, 3);
            TestLayer(layer);
        }

        [Fact]
        public void CanGradientCheckSoftmaxLayer()
        {
            var layer = new GruLayer<double>(5, 3);
            TestLayer(layer);
        }

        private void TestLayer(NeuroLayer<double> layer)
        {
            const double delta = 1e-5d;

            var dataSet = new TestDataSet<double>(layer.InputSize, layer.OutputSize, 5, 10);
            layer.Initialize(dataSet.BatchSize, dataSet.SampleCount);
            layer.InitSequence();

            var samples = dataSet.GetNextSamples(dataSet.SampleCount);

            for (int i = 0; i < samples.Inputs.Count; i++)
            {
                layer.Step(samples.Inputs[i], true);
            }

            layer.ErrorPropagate(samples.Targets);

            for (int i = 0; i < layer.TotalParamCount; i++)
            {
                var pLayer = layer.Clone();
                var nLayer = layer.Clone();

                pLayer.InitSequence();
                nLayer.InitSequence();

                pLayer.ResetMemory();
                nLayer.ResetMemory();

                pLayer.SetParam(i, pLayer.GetParam(i) + delta);
                nLayer.SetParam(i, nLayer.GetParam(i) - delta);

                double pErr = 0.0, nErr = 0.0;
                for (int j = 0; j < samples.Inputs.Count; j++)
                {
                    var pOut = pLayer.Step(samples.Inputs[j]);
                    var nOut = nLayer.Step(samples.Inputs[j]);

                    pErr += layer.LayerError(pOut, samples.Targets[j]);
                    nErr += layer.LayerError(nOut, samples.Targets[j]);
                }

                double num = (pErr - nErr) / (2.0f * delta);
                double real = layer.GetParam(i, true);
                double d = num - real;

                Math.Abs(d).ShouldBeLessThan(1e-7);
            }
        }
    }
}