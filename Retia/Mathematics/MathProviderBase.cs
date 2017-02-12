using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public abstract class MathProviderBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public abstract List<int> SoftMaxChoice(Matrix<T> p, double T = 1.0);

        protected abstract void ApplySigmoid2(IntPtr matrix1, IntPtr matrix2, int len);

        protected abstract void ApplyTanh(IntPtr matrix, int len);

        protected abstract double CrossEntropyInternal(T[] rawP, T[] rawT);

        protected abstract double MeanSquareInternal(T[] rawY, T[] rawT, out int notNan);

        protected abstract void SoftMaxNormInternal(T[] y, T[] result, int rows, int columns, double T);

        protected abstract Matrix<T> PropagateSingleError(Matrix<T> y, Matrix<T> target, int batchSize);

        public abstract T Scalar(float scalar);

        public abstract T Scalar(double scalar);

        protected abstract void CalculateHInternal(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int len);

        public void CalculateH(Matrix<T> H, Matrix<T> hCandidate, Matrix<T> z, Matrix<T> lastH)
        {
            using (var ptrs = new MatrixPointers<T>(H, hCandidate, z, lastH))
            {
                CalculateHInternal(ptrs[0], ptrs[1], ptrs[2], ptrs[3], H.Length());
            }
        }

        public List<Matrix<T>> ErrorPropagate(List<Matrix<T>> outputs, List<Matrix<T>> targets, int seqLen, int batchSize)
        {
            if (outputs.Count != targets.Count || targets.Count == 0)
                throw new Exception("Not enough targets provided or not enough output states stored!");

            var sensitivities = new List<Matrix<T>>(seqLen);
            
            for (int i = 0; i < seqLen; i++)
            {
                var y = outputs[i];
                var target = targets[i];
                sensitivities.Add(PropagateSingleError(y, target, batchSize));
            }

            return sensitivities;
        }


        public Matrix<T> SoftMaxNorm(Matrix<T> y, double T = 1.0)
        {
            var p = y.CloneMatrix();
            
            SoftMaxNormInternal(y.AsColumnMajorArray(), p.AsColumnMajorArray(), y.RowCount, y.ColumnCount, T);

            return p;
        }

        public void ApplySigmoid2(Matrix<T> matrix1, Matrix<T> matrix2)
        {
            using (var ptrs = new MatrixPointers<T>(matrix1, matrix2))
            {
                ApplySigmoid2(ptrs[0], ptrs[1], matrix1.Length());
            }
        }

        public void ApplyTanh(Matrix<T> matrix)
        {
            using (var ptrs = new MatrixPointers<T>(matrix))
            {
                ApplyTanh(ptrs[0], matrix.Length());
            }
        }

        public double CrossEntropy(Matrix<T> p, Matrix<T> target)
        {
            if (p.ColumnCount != target.ColumnCount || p.RowCount != target.RowCount)
                throw new Exception("Matrix dimensions must agree!");
            
            return -CrossEntropyInternal(p.AsColumnMajorArray(), target.AsColumnMajorArray()) / p.ColumnCount;
        }

        public double MeanSquare(Matrix<T> y, Matrix<T> target)
        {
            if (y.ColumnCount != target.ColumnCount || y.RowCount != target.RowCount)
                throw new Exception("Matrix dimensions must agree!");

            int notNan;
            double error = MeanSquareInternal(y.AsColumnMajorArray(), target.AsColumnMajorArray(), out notNan)

            return notNan == 0 ? 0.0 : 0.5 * error / notNan;
        }
    }
}