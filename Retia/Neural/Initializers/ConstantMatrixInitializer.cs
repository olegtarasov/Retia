using System;
using MathNet.Numerics.LinearAlgebra;

namespace Retia.Neural.Initializers
{
    public class ConstantMatrixInitializer<T> : IMatrixInitializer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public T Value { get; set; } = (T)Convert.ChangeType(0.0f, typeof(T));

        public Matrix<T> CreateMatrix(int rows, int columns)
        {
            return Matrix<T>.Build.Dense(rows, columns, Value);
        }
    }
}