using System;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;

namespace Retia.Neural.Initializers
{
    /// <summary>
    /// Creates matrices with all elements set to a specified value.
    /// </summary>
    public class ConstantMatrixInitializer<T> : IMatrixInitializer<T> where T : struct, IEquatable<T>, IFormattable
    {
        /// <summary>
        /// Value of all matrix elements.
        /// </summary>
        public T Value { get; set; } = MathProvider<T>.Instance.Scalar(0.0f);

        public Matrix<T> CreateMatrix(int rows, int columns)
        {
            return Matrix<T>.Build.Dense(rows, columns, Value);
        }
    }
}