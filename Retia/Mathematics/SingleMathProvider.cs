using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public class SingleMathProvider : MathProviderBase<float>
    {
        [DllImport("FastFuncs")]
        private static extern void ApplySigmoid2S(IntPtr a, IntPtr b, int n);

        [DllImport("FastFuncs")]
        private static extern void ApplyTanhS(IntPtr matrix, int n);

        [DllImport("FastFuncs")]
        private static extern void CalculateHS(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int n);

        [DllImport("FastFuncs")]
        private static extern void GravesRMSPropUpdateS(float weightDecay, float learningRate, float decayRate, float momentum, IntPtr weightMatrix, IntPtr grad1_cache, IntPtr grad2_cache, IntPtr momentum_cache, IntPtr gradient, int n);

        public override List<int> SoftMaxChoice(Matrix<float> p, double T = 1)
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

        protected override double CrossEntropyInternal(float[] rawP, float[] rawT)
        {
            //todo: should be fixed, we must take NaN cols into account

            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            double err = 0.0d;
            for (int i = 0; i < rawP.Length; i++)
            {
                if (float.IsNaN(rawT[i]))
                    continue;
                err += rawT[i] * Math.Log(rawP[i]);
            }

            return err;
        }

        protected override void ApplySigmoid2(IntPtr matrix1, IntPtr matrix2, int len)
        {
            ApplySigmoid2S(matrix1, matrix2, len);
        }

        protected override void ApplyTanh(IntPtr matrix, int len)
        {
            ApplyTanhS(matrix, len);
        }

        protected override double MeanSquareInternal(float[] rawY, float[] rawT, out int notNan)
        {
            //E(y0, ... ,yn) = 0.5/n(target0-y0)^2 + ... + 0.5/n(target_n - y_n)^2
            double err = 0.0d;
            notNan = 0;
            for (int i = 0; i < rawY.Length; i++)
            {
                if (float.IsNaN(rawT[i]))
                    continue;
                notNan++;
                double delta = rawT[i] - rawY[i];
                err += delta * delta;
            }

            return err;
        }

        protected override void SoftMaxNormInternal(float[] y, float[] result, int rows, int columns, double T)
        {
            var sums = new float[columns];
            for (int i = 0; i < y.Length; i++)
            {
                result[i] = (float)Math.Exp(result[i] / T);
                var c = i / rows;
                sums[c] += result[i];
            }

            for (int i = 0; i < y.Length; i++)
            {
                var c = i / rows;
                result[i] /= sums[c];
            }
        }

        protected override Matrix<float> PropagateSingleError(Matrix<float> y, Matrix<float> target, int batchSize)
        {
            return target.Map2((targetVal, yVal) => float.IsNaN(targetVal) ? 0.0f : yVal, y).Divide(batchSize);
        }

        public override float Scalar(float scalar)
        {
            return scalar;
        }

        public override float Scalar(double scalar)
        {
            return (float)scalar;
        }

        protected override void CalculateHInternal(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int len)
        {
            CalculateHS(H, hCandidate, z, lastH, len);
        }

        protected override void GravesRMSPropUpdateInternal(float weightDecay, float learningRate, float decayRate, float momentum, IntPtr weightMatrix, IntPtr grad1_cache, IntPtr grad2_cache, IntPtr momentum_cache, IntPtr gradient, int len)
        {
            GravesRMSPropUpdateS(weightDecay, learningRate, decayRate, momentum, weightMatrix, grad1_cache, grad2_cache, momentum_cache, gradient, len);
        }

        public override float NaN()
        {
            return float.NaN;
        }
    }
}