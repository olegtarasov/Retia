using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Mathematics;
using Retia.Optimizers;
using Retia.RandomGenerator;

namespace Retia.Neural.Layers
{
    public class SoftMaxLayer : NeuroLayer
    {
        private readonly int _size;

        private SoftMaxLayer(SoftMaxLayer other) : base(other)
        {
            _size = other._size;
        }

        public SoftMaxLayer(int size)
        {
            _size = size;
        }

        public SoftMaxLayer(BinaryReader reader)
        {
            _size = reader.ReadInt32();
        }

        public override int InputSize => _size;
        public override int OutputSize => _size;
        public override int TotalParamCount => 0;

        public override double LayerError(Matrix y, Matrix target)
        {
            return ErrorFunctions.CrossEntropy(y, target);
        }

        public override void Save(Stream s)
        {
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_size);
            }
        }

        public override NeuroLayer Clone()
        {
            return new SoftMaxLayer(this);
        }

        public override void Optimize(OptimizerBase optimizer)
        {
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            var output = SoftMax.SoftMaxNorm(input);
            if (inTraining)
            {
                Inputs.Add(input);
                Outputs.Add(output);
            }
            return output;
        }


        protected override double Derivative(Matrix input, Matrix output, int batch, int i, int o)
        {
            return i == o ? output[i, batch] * (1 - output[o, batch]) : -output[i, batch] * output[o, batch];
        }

        public override List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true)
        {
            if (Outputs.Count != Inputs.Count)
                throw new Exception("Backprop was not initialized (empty state sequence)");
            if (Inputs.Count == 0)
                throw new Exception("Empty inputs history, nothing to propagate!");
            if (outSens.Count != Inputs.Count)
                throw new Exception("Not enough sensitivies in list!");

            return PropagateSensitivity(outSens);
        }

        public override void ResetMemory()
        {
        }

        public override void ResetOptimizer()
        {
        }

        public override void InitSequence()
        {
            Outputs.Clear();
            Inputs.Clear();
        }

        public override void ClampGrads(float limit)
        {
        }

        public override void ToVectorState(double[] destination, ref int idx, bool grad = false)
        {
        }

        public override void FromVectorState(double[] vector, ref int idx)
        {
        }

        public override LayerSpecBase CreateSpec()
        {
            return new SoftmaxLayerSpec(_size, BatchSize, SeqLen);
        }
    }
}