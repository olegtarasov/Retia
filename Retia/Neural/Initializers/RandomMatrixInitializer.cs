using System;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;

namespace Retia.Neural.Initializers
{
    /// <summary>
    /// Creates random matrices with specified dispersion.
    /// </summary>
    public class RandomMatrixInitializer<T> : IMatrixInitializer<T> where T : struct, IEquatable<T>, IFormattable
    {
        /// <summary>
        /// Dispersion for random matrices.
        /// </summary>
        public float Dispersion { get; set; } = 5e-2f;

        public Matrix<T> CreateMatrix(int rows, int columns)
        {
            return MatrixFactory.RandomMatrix<T>(rows, columns, Dispersion);
        }
    }
}