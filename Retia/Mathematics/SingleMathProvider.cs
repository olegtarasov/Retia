using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural;
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

        public override float Scalar(float scalar)
        {
            return scalar;
        }

        public override float Scalar(double scalar)
        {
            return (float)scalar;
        }

        public override float NaN()
        {
            return float.NaN;
        }

        public override void GravesRmsPropUpdate(float weightDecay, float learningRate, float decayRate, float momentum, NeuroWeight<float> weight)
        {
            using (var ptrs = new MatrixPointers<float>(weight.Weight, weight.Cache1, weight.Cache2, weight.CacheM, weight.Gradient))
            {
                GravesRMSPropUpdateS(weightDecay, learningRate, decayRate, momentum, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], weight.Weight.Length());
            }
        }

        public override void CalculateH(Matrix<float> H, Matrix<float> hCandidate, Matrix<float> z, Matrix<float> lastH)
        {
            using (var ptrs = new MatrixPointers<float>(H, hCandidate, z, lastH))
            {
                CalculateHS(ptrs[0], ptrs[1], ptrs[2], ptrs[3], H.Length());
            }
        }

        public override Matrix<float> SoftMaxNorm(Matrix<float> y, double T = 1)
        {
            var p = y.CloneMatrix();

            var ya = y.AsColumnMajorArray();
            var pa = p.AsColumnMajorArray();

            var sums = new float[y.ColumnCount];
            for (int i = 0; i < ya.Length; i++)
            {
                pa[i] = (float)Math.Exp(pa[i] / T);
                var c = i / y.RowCount;
                sums[c] += pa[i];
            }

            for (int i = 0; i < ya.Length; i++)
            {
                var c = i / y.RowCount;
                pa[i] /= sums[c];
            }

            return p;
        }

        public override void ApplySigmoid2(Matrix<float> matrix1, Matrix<float> matrix2)
        {
            using (var ptrs = new MatrixPointers<float>(matrix1, matrix2))
            {
                ApplySigmoid2S(ptrs[0], ptrs[1], matrix1.Length());
            }
        }

        public override void ApplyTanh(Matrix<float> matrix)
        {
            using (var ptrs = new MatrixPointers<float>(matrix))
            {
                ApplyTanhS(ptrs[0], matrix.Length());
            }
        }

        public override double CrossEntropy(Matrix<float> p, Matrix<float> target)
        {
            if (p.ColumnCount != target.ColumnCount || p.RowCount != target.RowCount)
                throw new Exception("Matrix dimensions must agree!");

            var pa = p.AsColumnMajorArray();
            var ta = target.AsColumnMajorArray();

            //todo: should be fixed, we must take NaN cols into account
            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            double err = 0.0d;
            for (int i = 0; i < pa.Length; i++)
            {
                if (double.IsNaN(ta[i]))
                    continue;
                err += ta[i] * Math.Log(pa[i]);
            }

            return -err / p.ColumnCount;
        }

        public override double MeanSquare(Matrix<float> y, Matrix<float> target)
        {
            if (y.ColumnCount != target.ColumnCount || y.RowCount != target.RowCount)
                throw new Exception("Matrix dimensions must agree!");

            var ya = y.AsColumnMajorArray();
            var ta = target.AsColumnMajorArray();

            //E(y0, ... ,yn) = 0.5/n(target0-y0)^2 + ... + 0.5/n(target_n - y_n)^2
            int notNan;
            double err = 0.0d;
            notNan = 0;
            for (int i = 0; i < ya.Length; i++)
            {
                if (double.IsNaN(ta[i]))
                    continue;
                notNan++;
                double delta = ta[i] - ya[i];
                err += delta * delta;
            }

            return notNan == 0 ? 0.0 : 0.5 * err / notNan;
        }

        protected override Matrix<float> PropagateSingleError(Matrix<float> y, Matrix<float> target, int batchSize)
        {
            return target.Map2((targetVal, yVal) => float.IsNaN(targetVal) ? 0.0f : yVal, y).Divide(batchSize);
        }

        protected override bool AlmostEqual(float a, float b)
        {
            return Math.Abs(a - b) < 10e-7f;
        }

        public override float[] Array(params float[] input)
        {
            return input;
        }

        public override void ClampMatrix(Matrix<float> matrix, float min, float max)
        {
            var arr = matrix.AsColumnMajorArray();
            for (int i = 0; i < arr.Length; i++)
            {
                if (arr[i] > max)
                    arr[i] = max;
                if (arr[i] < min)
                    arr[i] = min;
            }
        }

        public override Matrix<float> RandomMatrix(int rows, int cols, float min, float max)
        {
            var random = SafeRandom.Generator;
            var arr = new float[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (float)random.NextDouble(min, max);
            return Matrix<float>.Build.Dense(rows, cols, arr);
        }
    }
}