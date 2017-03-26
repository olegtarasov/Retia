using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Mathematics;
using Retia.Neural.ErrorFunctions;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class TanhLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _size;

        private TanhLayer(TanhLayer<T> other) : base(other)
        {
            _size = other._size;
        }

        public TanhLayer(int size)
        {
            _size = size;

            ErrorFunction = new MeanSquareError<T>();
        }

        public TanhLayer(BinaryReader reader) : base(reader)
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
            return new TanhLayer<T>(_size);
        }

        public override void Optimize(OptimizerBase<T> optimizer)
        {
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            var output = input.CloneMatrix();

            MathProvider.ApplyTanh(output);

            if (inTraining)
            {
                Outputs.Add(output);
            }

            return output;
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true, bool clearGrad = true)
        {
            if (Outputs.Count == 0)
                throw new Exception("Empty inputs history, nothing to propagate!");
            if (outSens.Count != Outputs.Count)
                throw new Exception("Not enough sensitivies in list!");

            var ones = Matrix<T>.Build.Dense(outSens[0].RowCount, outSens[0].ColumnCount, Matrix<T>.One);
            var iSens = new List<Matrix<T>>();
            for (int t = 0; t < outSens.Count; t++)
            {
                var output = Outputs[t];
                var osens = outSens[t];
                
                // osens ^ (1 - tanh(x)^2)
                iSens.Add((ones - output.PointwiseMultiply(output)).PointwiseMultiply(osens));
            }
            return iSens;
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
            throw new NotSupportedException();
        }

        public override void TransferWeightsToDevice()
        {
            throw new NotSupportedException();
        }

        public override void TransferWeightsToHost()
        {
            throw new NotSupportedException();
        }
    }
}
