﻿using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;
using Retia.RandomGenerator;

namespace Retia.Neural.Initializers
{
    public class ProportionalRandomMatrixInitializer : IMatrixInitializer
    {
        public double Dispersion { get; set; } = 5e-2;

        public Matrix CreateMatrix(int rows, int columns)
        {
            return DenseMatrix.CreateRandom(rows, columns, new Normal(0.0f, Dispersion / columns));
        }

        public float[] CreateArray(int size)
        {
            return RandomMatrixInitializer.CreateRandomArray(size, Dispersion);
        }
    }
}