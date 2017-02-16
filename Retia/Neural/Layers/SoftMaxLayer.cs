using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Mathematics;
using Retia.Neural.ErrorFunctions;
using Retia.Optimizers;
using Retia.RandomGenerator;

namespace Retia.Neural.Layers
{
    public class SoftMaxLayer<T> : NeuroLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _size;

        private SoftMaxLayer(SoftMaxLayer<T> other) : base(other)
        {
            _size = other._size;
        }

        public SoftMaxLayer(int size)
        {
            _size = size;

            ErrorFunction = new CrossEntropyError<T>();
        }

        public SoftMaxLayer(BinaryReader reader) : base(reader)
        {
            _size = reader.ReadInt32();
        }

        public override int InputSize => _size;
        public override int OutputSize => _size;
        public override int TotalParamCount => 0;

        public override void Save(Stream s)
        {
            base.Save(s);

            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_size);
            }
        }

        public override NeuroLayer<T> Clone()
        {
            return new SoftMaxLayer<T>(this);
        }

        public override void Optimize(OptimizerBase<T> optimizer)
        {
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            var output = MathProvider.SoftMaxNorm(input);
            if (inTraining)
            {
                Inputs.Add(input);
                Outputs.Add(output);
            }
            return output;
        }


        protected override T Derivative(Matrix<T> input, Matrix<T> output, int batch, int i, int o)
        {
            // TODO: Support
            throw new NotSupportedException();
            //return i == o ? output[i, batch] * (1 - output[o, batch]) : -output[i, batch] * output[o, batch];
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true)
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

        public override void ToVectorState(T[] destination, ref int idx, bool grad = false)
        {
        }

        public override void FromVectorState(T[] vector, ref int idx)
        {
        }

        public override LayerSpecBase CreateSpec()
        {
            return new SoftmaxLayerSpec(_size, BatchSize, SeqLen);
        }
    }
}