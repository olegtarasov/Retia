using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public static class SoftMax
    {
        public static Matrix SoftMaxNorm(Matrix y, double T = 1.0)
        {
            var p = y.CloneMatrix();
            var rawP = p.AsColumnMajorArray();
            var n = rawP.Length;
            var sums = new double[y.ColumnCount];
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

        public static List<int> SoftMaxChoice(Matrix p, double T = 1.0)
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