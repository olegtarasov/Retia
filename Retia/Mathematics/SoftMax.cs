using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public static class SoftMax<T> where T : struct, IEquatable<T>, IFormattable
    {
        private static readonly bool _isFloat = typeof(T) == typeof(float);

        public static Matrix<T> SoftMaxNorm(Matrix<T> y, double T = 1.0)
        {
            if (_isFloat)
            {
                return SoftMaxNorm(y as Matrix<float>, T) as Matrix<T>;
            }

            return SoftMaxNorm(y as Matrix<double>, T) as Matrix<T>;
        }

        public static Matrix<float> SoftMaxNorm(Matrix<float> y, double T = 1.0)
        {
            var p = y.CloneMatrix();
            var rawP = p.AsColumnMajorArray();
            var n = rawP.Length;
            var sums = new float[y.ColumnCount];
            for (int i = 0; i < n; i++)
            {
                rawP[i] = (float)Math.Exp(rawP[i] / T);
                var c = i / y.RowCount;
                sums[c] += rawP[i];
            }

            for (int i = 0; i < n; i++)
            {
                var c = i / y.RowCount;
                rawP[i] /= sums[c];
            }
            return p;
        }

        public static Matrix<double> SoftMaxNorm(Matrix<double> y, double T = 1.0)
        {
            var p = y.CloneMatrix();
            var rawP = p.AsColumnMajorArray();
            var n = rawP.Length;
            var sums = new double[y.ColumnCount];
            for (int i = 0; i < n; i++)
            {
                rawP[i] = Math.Exp(rawP[i] / T);
                var c = i / y.RowCount;
                sums[c] += rawP[i];
            }

            for (int i = 0; i < n; i++)
            {
                var c = i / y.RowCount;
                rawP[i] /= sums[c];
            }
            return p;
        }

        public static List<int> SoftMaxChoice(Matrix<T> p, double T = 1.0)
        {
            if (_isFloat)
            {
                return SoftMaxChoice(p as Matrix<float>, T);
            }
            else
            {
                return SoftMaxChoice(p as Matrix<double>, T);
            }
        }

        public static List<int> SoftMaxChoice(Matrix<float> p, double T = 1.0)
        {
            var probs = new List<int>(p.ColumnCount);
            var rnd = SafeRandom.Generator;

            for (int j = 0; j < p.ColumnCount; j++)
            {
                var dChoice = rnd.NextDouble();
                double curPos = 0;
                double nextPos = p[0, j];

                int i;
                for (i = 1; i < p.ColumnCount; i++)
                {
                    if (dChoice > curPos && dChoice <= nextPos)
                        break;
                    curPos = nextPos;
                    nextPos += p[i, j];
                }

                probs.Add(i - 1);
            }
            return probs;
        }

        public static List<int> SoftMaxChoice(Matrix<double> p, double T = 1.0)
        {
            var probs = new List<int>(p.ColumnCount);
            var rnd = SafeRandom.Generator;

            for (int j = 0; j < p.ColumnCount; j++)
            {
                var dChoice = rnd.NextDouble();
                double curPos = 0;
                double nextPos = p[0, j];

                int i;
                for (i = 1; i < p.ColumnCount; i++)
                {
                    if (dChoice > curPos && dChoice <= nextPos)
                        break;
                    curPos = nextPos;
                    nextPos += p[i, j];
                }

                probs.Add(i - 1);
            }
            return probs;
        }
    }
}