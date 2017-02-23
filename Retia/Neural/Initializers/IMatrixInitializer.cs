using System;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;

namespace Retia.Neural.Initializers
{
    /// <summary>
    /// An generic interface for matrix creation.
    /// </summary>
    public interface IMatrixInitializer<T> where T : struct, IEquatable<T>, IFormattable
    {
        /// <summary>
        /// Creates a new matrix.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="columns">Number of columns.</param>
        Matrix<T> CreateMatrix(int rows, int columns);
    }
}