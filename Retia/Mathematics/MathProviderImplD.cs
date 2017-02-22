
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural;
using Retia.RandomGenerator;

using Float = System.Double;

namespace Retia.Mathematics
{
    /// <summary>
    /// Math provider for the float type. Double-precision is generated from the single-precision source.
    /// </summary>
    public class MathProviderImplD : MathProviderBase<Float>
    {
        [DllImport("FastFuncs", EntryPoint="ApplySigmoid2D")] private static extern void ApplySigmoid2(IntPtr a, IntPtr b, int n);

        [DllImport("FastFuncs", EntryPoint="ApplyTanhD")] private static extern void ApplyTanh(IntPtr matrix, int n);

        [DllImport("FastFuncs", EntryPoint="CalculateHD")] private static extern void CalculateH(IntPtr H, IntPtr hCandidate, IntPtr z, IntPtr lastH, int n);

        [DllImport("FastFuncs", EntryPoint="GravesRMSPropUpdateD")] private static extern void GravesRMSPropUpdate(Float weightDecay, Float learningRate, Float decayRate, Float momentum, IntPtr weightMatrix, IntPtr grad1_cache, IntPtr grad2_cache, IntPtr momentum_cache, IntPtr gradient, int n);

        [DllImport("FastFuncs", EntryPoint = "AdagradUpdateD")] private static extern void AdagradUpdate(Float learningRate, IntPtr weightMatrix, IntPtr mem, IntPtr gradient, int n);

        public override void AdagradUpdate(Float learningRate, NeuroWeight<Float> weight)
        {
            using (var ptrs = new MatrixPointers<Float>(weight.Weight, weight.Cache2, weight.Gradient))
            {
                AdagradUpdate(learningRate, ptrs[0], ptrs[1], ptrs[2], weight.Weight.Length());
            }
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

        public override Float[] Array(params float[] input)
        {
            return input.Select(x => (Float)x).ToArray();
        }

        public override Matrix<Float> BackPropagateCrossEntropyError(Matrix<Float> output, Matrix<Float> target)
        {
            var result = Matrix<Float>.Build.Dense(output.RowCount, output.ColumnCount);
            var oa = output.AsColumnMajorArray();
            var ta = target.AsColumnMajorArray();
            var ra = result.AsColumnMajorArray();

            int rows = output.RowCount;
            int cols = output.ColumnCount;
            int notNan = 0;
            for (int i = 0; i < oa.Length; i++)
            {
                if (i > 0 && i % rows == 0)
                {
                    if (notNan == 0)
                        cols--;

                    notNan = 0;
                }

                if (Float.IsNaN(ta[i]))
                {
                    continue;
                }

                notNan++;

                ra[i] = oa[i] - ta[i];
            }

            if (cols == 0)
            {
                throw new InvalidOperationException("All of your targets are NaN! This is pointless.");
            }

            for (int i = 0; i < ra.Length; i++)
            {
                ra[i] /= cols;
            }

            return result;
        }

        public override Matrix<Float> BackPropagateMeanSquareError(Matrix<Float> output, Matrix<Float> target)
        {
            var result = Matrix<Float>.Build.Dense(output.RowCount, output.ColumnCount);
            var oa = output.AsColumnMajorArray();
            var ta = target.AsColumnMajorArray();
            var ra = result.AsColumnMajorArray();

            int rows = output.RowCount;
            int notNan = 0;
            for (int i = 0; i < oa.Length; i++)
            {
                if (Float.IsNaN(ta[i]))
                {
                    continue;
                }

                notNan++;

                ra[i] = oa[i] - ta[i];
            }

            for (int i = 0; i < ra.Length; i++)
            {
                ra[i] /= notNan;
            }

            return result;
        }

        public override void CalculateH(Matrix<Float> H, Matrix<Float> hCandidate, Matrix<Float> z, Matrix<Float> lastH)
        {
            using (var ptrs = new MatrixPointers<Float>(H, hCandidate, z, lastH))
            {
                CalculateH(ptrs[0], ptrs[1], ptrs[2], ptrs[3], H.Length());
            }
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

        public override double CrossEntropyError(Matrix<Float> p, Matrix<Float> target)
        {
            if (p.ColumnCount != target.ColumnCount || p.RowCount != target.RowCount)
                throw new Exception("Matrix dimensions must agree!");

            var pa = p.AsColumnMajorArray();
            var ta = target.AsColumnMajorArray();

            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            double err = 0.0d;
            int notNan = 0;
            int cols = p.ColumnCount;
            for (int i = 0; i < pa.Length; i++)
            {
                if (i > 0 && i % p.RowCount == 0)
                {
                    if (notNan == 0)
                        cols--;

                    notNan = 0;
                }

                if (Float.IsNaN(ta[i]))
                    continue;

                notNan++;
                
                err += ta[i] * Math.Log(pa[i]);
            }

            if (cols == 0)
            {
                throw new InvalidOperationException("All of your targets are NaN! This is pointless.");
            }

            return -err / cols;
        }

        public override void GravesRmsPropUpdate(float weightDecay, float learningRate, float decayRate, float momentum, NeuroWeight<Float> weight)
        {
            using (var ptrs = new MatrixPointers<Float>(weight.Weight, weight.Cache1, weight.Cache2, weight.CacheM, weight.Gradient))
            {
                GravesRMSPropUpdate(weightDecay, learningRate, decayRate, momentum, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], weight.Weight.Length());
            }
        }

        public override double MeanSquareError(Matrix<Float> y, Matrix<Float> target)
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

        public override Float NaN()
        {
            return Float.NaN;
        }

        public override Matrix<Float> RandomMatrix(int rows, int cols, float min, float max)
        {
            var random = SafeRandom.Generator;
            var arr = new Float[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (Float)random.NextDouble(min, max);
            return Matrix<Float>.Build.Dense(rows, cols, arr);
        }

        public override Float Scalar(float scalar)
        {
            return (Float)scalar;
        }

        public override Float Scalar(double scalar)
        {
            return (Float)scalar;
        }

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
                for (i = 1; i < p.RowCount; i++)
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

        protected override bool AlmostEqual(Float a, Float b)
        {
            return Math.Abs(a - b) < 10e-7f;
        }
    }
}
