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
            
        }
#endif
    }
}