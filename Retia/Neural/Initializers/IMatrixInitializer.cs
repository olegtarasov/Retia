using System;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;

namespace Retia.Neural.Initializers
{
    public interface IMatrixInitializer<T> where T : struct, IEquatable<T>, IFormattable
    {
        Matrix<T> CreateMatrix(int rows, int columns);
    }
}