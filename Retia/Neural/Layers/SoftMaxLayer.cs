using System;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Retia.Interop;
using Retia.Helpers;
using Retia.Integration.Helpers;
using Retia.Neural.ErrorFunctions;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class SoftMaxLayer<T> : DerivativeLayerBase<T> where T : struct, IEquatable<T>, IFormattable
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

        public override LayerBase<T> Clone()
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

        protected override double DerivativeD(Matrix<double> input, Matrix<double> output, int batch, int i, int o)
        {
            return i == o ? output[i, batch] * (1 - output[o, batch]) : -output[i, batch] * output[o, batch];
        }

        protected override float DerivativeS(Matrix<float> input, Matrix<float> output, int batch, int i, int o)
        {
            return i == o ? output[i, batch] * (1 - output[o, batch]) : -output[i, batch] * output[o, batch];
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

        public override void ClearGradients()
        {
        }

        public override IntPtr CreateGpuLayer()
        {
            GpuLayerPtr = GpuInterface.CreateSoftmaxLayer(_size, BatchSize, SeqLen);

            return GpuLayerPtr;
        }

        public override void TransferWeightsToDevice()
        {
        }

        public override void TransferWeightsToHost()
        {
        }
    }
}