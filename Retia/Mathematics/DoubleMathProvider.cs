using FP = System.Double;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public class DoubleMathProvider : MathProviderBase<FP>
    {
        [DllImport("FastFuncs", EntryPoint="ApplySigmoid2D")] private static extern void ApplySigmoid2(IntPtr a, IntPtr b, int n);

        [DllImport("FastFuncs", EntryPoint="ApplyTanhD")] private static extern void ApplyTanh(IntPtr matrix, int n);

        [DllImport("FastFuncs", EntryPoint="CalculateHD")] private static extern void CalculateH(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int n);

        [DllImport("FastFuncs", EntryPoint="GravesRMSPropUpdateD")] private static extern void GravesRMSPropUpdate(FP weightDecay, FP learningRate, FP decayRate, FP momentum, IntPtr weightMatrix, IntPtr grad1_cache, IntPtr grad2_cache, IntPtr momentum_cache, IntPtr gradient, int n);

        public override List<int> SoftMaxChoice(Matrix<FP> p, double T = 1)
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

        public override FP Scalar(float scalar)
        {
            return (FP)scalar;
        }

        public override FP Scalar(double scalar)
        {
            return (FP)scalar;
        }

        public override FP NaN()
        {
            return FP.NaN;
        }

        public override void GravesRmsPropUpdate(float weightDecay, float learningRate, float decayRate, float momentum, NeuroWeight<FP> weight)
        {
            using (var ptrs = new MatrixPointers<FP>(weight.Weight, weight.Cache1, weight.Cache2, weight.CacheM, weight.Gradient))
            {
                GravesRMSPropUpdate(weightDecay, learningRate, decayRate, momentum, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], weight.Weight.Length());
            }
        }

        public override void CalculateH(Matrix<FP> H, Matrix<FP> hCandidate, Matrix<FP> z, Matrix<FP> lastH)
        {
            using (var ptrs = new MatrixPointers<FP>(H, hCandidate, z, lastH))
            {
                CalculateH(ptrs[0], ptrs[1], ptrs[2], ptrs[3], H.Length());
            }
        }

        public override Matrix<FP> SoftMaxNorm(Matrix<FP> y, double T = 1)
        {
            var p = y.CloneMatrix();

            var ya = y.AsColumnMajorArray();
            var pa = p.AsColumnMajorArray();

            var sums = new FP[y.ColumnCount];
            for (int i = 0; i < ya.Length; i++)
            {
                pa[i] = (FP)Math.Exp(pa[i] / T);
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

        public override void ApplySigmoid2(Matrix<FP> matrix1, Matrix<FP> matrix2)
        {
            using (var ptrs = new MatrixPointers<FP>(matrix1, matrix2))
            {
                ApplySigmoid2(ptrs[0], ptrs[1], matrix1.Length());
            }
        }

        public override void ApplyTanh(Matrix<FP> matrix)
        {
            using (var ptrs = new MatrixPointers<FP>(matrix))
            {
                ApplyTanh(ptrs[0], matrix.Length());
            }
        }

        public override double CrossEntropy(Matrix<FP> p, Matrix<FP> target)
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
                if (FP.IsNaN(ta[i]))
                    continue;
                err += ta[i] * Math.Log(pa[i]);
            }

            return -err / p.ColumnCount;
        }

        public override double MeanSquare(Matrix<FP> y, Matrix<FP> target)
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
                if (FP.IsNaN(ta[i]))
                    continue;
                notNan++;
                double delta = ta[i] - ya[i];
                err += delta * delta;
            }

            return notNan == 0 ? 0.0 : 0.5 * err / notNan;
        }

        protected override Matrix<FP> PropagateSingleError(Matrix<FP> y, Matrix<FP> target, int batchSize)
        {
            return target.Map2((targetVal, yVal) => FP.IsNaN(targetVal) ? (FP)0.0f : yVal - targetVal, y).Divide(batchSize);
        }

        protected override bool AlmostEqual(FP a, FP b)
        {
            return Math.Abs(a - b) < 10e-7f;
        }

        public override FP[] Array(params float[] input)
        {
            return input.Select(x => (FP)x).ToArray();
        }

        public override void ClampMatrix(Matrix<FP> matrix, FP min, FP max)
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

        public override Matrix<FP> RandomMatrix(int rows, int cols, float min, float max)
        {
            var random = SafeRandom.Generator;
            var arr = new FP[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (FP)random.NextDouble(min, max);
            return Matrix<FP>.Build.Dense(rows, cols, arr);
        }
    }
}
