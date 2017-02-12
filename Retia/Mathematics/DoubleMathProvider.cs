using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public class DoubleMathProvider : MathProviderBase<double>
    {
        [DllImport("FastFuncs")]
        private static extern void ApplySigmoid2D(IntPtr a, IntPtr b, int n);

        [DllImport("FastFuncs")]
        private static extern void ApplyTanhD(IntPtr matrix, int n);

        [DllImport("FastFuncs")]
        private static extern void CalculateHD(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int n);

        public override List<int> SoftMaxChoice(Matrix<double> p, double T = 1)
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

        protected override double CrossEntropyInternal(double[] rawP, double[] rawT)
        {
            //todo: should be fixed, we must take NaN cols into account

            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            double err = 0.0d;
            for (int i = 0; i < rawP.Length; i++)
            {
                if (double.IsNaN(rawT[i]))
                    continue;
                err += rawT[i] * Math.Log(rawP[i]);
            }

            return err;
        }

        protected override void ApplySigmoid2(IntPtr matrix1, IntPtr matrix2, int len)
        {
            ApplySigmoid2D(matrix1, matrix2, len);
        }

        protected override void ApplyTanh(IntPtr matrix, int len)
        {
            ApplyTanhD(matrix, len);
        }

        protected override double MeanSquareInternal(double[] rawY, double[] rawT, out int notNan)
        {
            //E(y0, ... ,yn) = 0.5/n(target0-y0)^2 + ... + 0.5/n(target_n - y_n)^2
            double err = 0.0d;
            notNan = 0;
            for (int i = 0; i < rawY.Length; i++)
            {
                if (double.IsNaN(rawT[i]))
                    continue;
                notNan++;
                double delta = rawT[i] - rawY[i];
                err += delta * delta;
            }

            return err;
        }

        protected override void SoftMaxNormInternal(double[] y, double[] result, int rows, int columns, double T)
        {
            var sums = new double[columns];
            for (int i = 0; i < y.Length; i++)
            {
                result[i] = Math.Exp(result[i] / T);
                var c = i / rows;
                sums[c] += result[i];
            }

            for (int i = 0; i < y.Length; i++)
            {
                var c = i / rows;
                result[i] /= sums[c];
            }
        }

        protected override Matrix<double> PropagateSingleError(Matrix<double> y, Matrix<double> target, int batchSize)
        {
            return target.Map2((targetVal, yVal) => double.IsNaN(targetVal) ? 0.0d : yVal, y).Divide(batchSize);
        }

        public override double Scalar(float scalar)
        {
            return (double)scalar;
        }

        public override double Scalar(double scalar)
        {
            return scalar;
        }

        protected override void CalculateHInternal(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int len)
        {
            CalculateHD(H, hCandidate, z, lastH, len);
        }
    }
}