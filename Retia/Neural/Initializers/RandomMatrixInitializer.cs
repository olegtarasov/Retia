﻿using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;
using Retia.RandomGenerator;

namespace Retia.Neural.Initializers
{
    public class RandomMatrixInitializer : IMatrixInitializer
    {
        public double Dispersion { get; set; } = 5e-2;

        public Matrix CreateMatrix(int rows, int columns)
        {
            return DenseMatrix.CreateRandom(rows, columns, new Normal(0.0f, Dispersion));
        }

        public float[] CreateArray(int size)
        {
            return CreateRandomArray(size, Dispersion);
        }

        public static float[] CreateRandomArray(int size, double dispersion)
        {
            var rnd = SafeRandom.Generator;
            var result = new float[size];

            for (int i = 0; i < size; i++)
            {
                result[i] = (float)rnd.NextDouble(-dispersion, dispersion);
            }

            return result;
        }
    }
}