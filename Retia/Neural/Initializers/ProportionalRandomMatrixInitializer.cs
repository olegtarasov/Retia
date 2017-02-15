using System;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;
using Retia.RandomGenerator;

namespace Retia.Neural.Initializers
{
    public class ProportionalRandomMatrixInitializer<T> : IMatrixInitializer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public double Dispersion { get; set; } = 5e-2;

        public Matrix<T> CreateMatrix(int rows, int columns)
        {
            return Matrix<T>.Build.Random(rows, columns, new ContinuousUniform(0.0f, Dispersion / columns));
        }
    }
}