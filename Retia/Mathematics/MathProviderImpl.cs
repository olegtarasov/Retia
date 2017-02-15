﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural;
using Retia.RandomGenerator;

using Float = System.Single;

namespace Retia.Mathematics
{
    public class MathProviderImpl : MathProviderBase<Float>
    {
        [DllImport("FastFuncs", EntryPoint="ApplySigmoid2S")] private static extern void ApplySigmoid2(IntPtr a, IntPtr b, int n);

        [DllImport("FastFuncs", EntryPoint="ApplyTanhS")] private static extern void ApplyTanh(IntPtr matrix, int n);

        [DllImport("FastFuncs", EntryPoint="CalculateHS")] private static extern void CalculateH(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int n);

        [DllImport("FastFuncs", EntryPoint="GravesRMSPropUpdateS")] private static extern void GravesRMSPropUpdate(Float weightDecay, Float learningRate, Float decayRate, Float momentum, IntPtr weightMatrix, IntPtr grad1_cache, IntPtr grad2_cache, IntPtr momentum_cache, IntPtr gradient, int n);

        public override List<int> SoftMaxChoice(Matrix<Float> p, double T = 1)
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

        public override Float Scalar(float scalar)
        {
            return (Float)scalar;
        }

        public override Float Scalar(double scalar)
        {
            return (Float)scalar;
        }

        public override Float NaN()
        {
            return Float.NaN;
        }

        public override void GravesRmsPropUpdate(float weightDecay, float learningRate, float decayRate, float momentum, NeuroWeight<Float> weight)
        {
            using (var ptrs = new MatrixPointers<Float>(weight.Weight, weight.Cache1, weight.Cache2, weight.CacheM, weight.Gradient))
            {
                GravesRMSPropUpdate(weightDecay, learningRate, decayRate, momentum, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], weight.Weight.Length());
            }
        }

        public override void CalculateH(Matrix<Float> H, Matrix<Float> hCandidate, Matrix<Float> z, Matrix<Float> lastH)
        {
            using (var ptrs = new MatrixPointers<Float>(H, hCandidate, z, lastH))
            {
                CalculateH(ptrs[0], ptrs[1], ptrs[2], ptrs[3], H.Length());
            }
        }

        public override Matrix<Float> SoftMaxNorm(Matrix<Float> y, double T = 1)
        {
            var p = y.CloneMatrix();

            var ya = y.AsColumnMajorArray();
            var pa = p.AsColumnMajorArray();

            var sums = new Float[y.ColumnCount];
            for (int i = 0; i < ya.Length; i++)
            {
                pa[i] = (Float)Math.Exp(pa[i] / T);
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

        public override void ApplySigmoid2(Matrix<Float> matrix1, Matrix<Float> matrix2)
        {
            using (var ptrs = new MatrixPointers<Float>(matrix1, matrix2))
            {
                ApplySigmoid2(ptrs[0], ptrs[1], matrix1.Length());
            }
        }

        public override void ApplyTanh(Matrix<Float> matrix)
        {
            using (var ptrs = new MatrixPointers<Float>(matrix))
            {
                ApplyTanh(ptrs[0], matrix.Length());
            }
        }

        public override double CrossEntropy(Matrix<Float> p, Matrix<Float> target)
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
                if (Float.IsNaN(ta[i]))
                    continue;
                err += ta[i] * Math.Log(pa[i]);
            }

            return -err / p.ColumnCount;
        }

        public override double MeanSquare(Matrix<Float> y, Matrix<Float> target)
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
                if (Float.IsNaN(ta[i]))
                    continue;
                notNan++;
                double delta = ta[i] - ya[i];
                err += delta * delta;
            }

            return notNan == 0 ? 0.0 : 0.5 * err / notNan;
        }

        protected override Matrix<Float> PropagateSingleError(Matrix<Float> y, Matrix<Float> target, int batchSize)
        {
            return target.Map2((targetVal, yVal) => Float.IsNaN(targetVal) ? (Float)0.0f : yVal - targetVal, y).Divide(batchSize);
        }

        protected override bool AlmostEqual(Float a, Float b)
        {
            return Math.Abs(a - b) < 10e-7f;
        }

        public override Float[] Array(params float[] input)
        {
            return input.Select(x => (Float)x).ToArray();
        }

        public override void ClampMatrix(Matrix<Float> matrix, Float min, Float max)
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

        public override Matrix<Float> RandomMatrix(int rows, int cols, float min, float max)
        {
            var random = SafeRandom.Generator;
            var arr = new Float[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (Float)random.NextDouble(min, max);
            return Matrix<Float>.Build.Dense(rows, cols, arr);
        }
    }
}