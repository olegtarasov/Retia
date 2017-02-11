using System;
using System.Collections.Generic;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.Providers.LinearAlgebra;

namespace Retia.Mathematics
{
    public static class MatrixExtensions
    {
        public static int Length<T>(this Matrix<T> matrix) where T : struct, IEquatable<T>, IFormattable
        {
            return matrix.RowCount * matrix.ColumnCount;
        }

        public static Matrix<T> CloneMatrix<T>(this Matrix<T> matrix) where T : struct, IEquatable<T>, IFormattable
        {
            return Matrix<T>.Build.DenseOfMatrix(matrix);
        }

        public static void Clamp<T>(this Matrix<T> matrix, T min, T max) where T : struct, IEquatable<T>, IFormattable
        {
            matrix.PointwiseMaximum(max);
            matrix.PointwiseMinimum(min);
            //var arr = matrix.AsColumnMajorArray();
            //for (int i = 0; i < arr.Length; i++)
            //{
            //    var value = Convert.ToSingle(arr[i]);
            //    if (value < min)
            //        arr[i] = min;
            //    else if (value > max)
            //        arr[i] = max;
            //}
        }

        public static Matrix<T> TileColumns<T>(this Matrix<T> matrix, int count) where T : struct, IEquatable<T>, IFormattable
        {
            var src = matrix.AsColumnMajorArray();
            var dst = new T[src.Length * count];

            for (int i = 0; i < count; i++)
            {
                Array.Copy(src, 0, dst, src.Length * i, src.Length);
            }

            return Matrix<T>.Build.Dense(matrix.RowCount, matrix.ColumnCount * count, dst);
        }

        public static Matrix<T> TileRows<T>(this Matrix<T> matrix, int count) where T : struct, IEquatable<T>, IFormattable
        {
            var src = matrix.AsColumnMajorArray();
            var dst = new T[src.Length * count];

            for (int col = 0; col < matrix.ColumnCount; col++)
            {
                for (int row = 0; row < count; row++)
                {
                    Array.Copy(src, col * matrix.RowCount, dst, matrix.RowCount * col * count + matrix.RowCount * row, matrix.RowCount);
                }
            }

            return Matrix<T>.Build.Dense(matrix.RowCount * count, matrix.ColumnCount, dst);
        }

        /// <summary>
        ///     this = beta*this + alpha*AB;
        /// </summary>
        public static void Accumulate(this Matrix<float> C, Matrix<float> A, Matrix<float> B, float beta = 0.0f, float alpha = 1.0f, Transpose transposeA = Transpose.DontTranspose, Transpose transposeB = Transpose.DontTranspose) 
        {
            Control.LinearAlgebraProvider.MatrixMultiplyWithUpdate(transposeA, transposeB, alpha, A.AsColumnMajorArray(), A.RowCount, A.ColumnCount, B.AsColumnMajorArray(), B.RowCount, B.ColumnCount, beta, C.AsColumnMajorArray());
        }

        /// <summary>
        ///     this = beta*this + alpha*AB;
        /// </summary>
        public static void Accumulate(this Matrix<double> C, Matrix<double> A, Matrix<double> B, double beta = 0.0d, double alpha = 1.0d, Transpose transposeA = Transpose.DontTranspose, Transpose transposeB = Transpose.DontTranspose) 
        {
            Control.LinearAlgebraProvider.MatrixMultiplyWithUpdate(transposeA, transposeB, alpha, A.AsColumnMajorArray(), A.RowCount, A.ColumnCount, B.AsColumnMajorArray(), B.RowCount, B.ColumnCount, beta, C.AsColumnMajorArray());
        }

        public static void CollapseColumnsAndAccumulate(this Matrix<float> C, Matrix<float> A, Matrix<float> B)
        {
            if (A.ColumnCount > 1)
            {
                C.Accumulate(A, B, 1.0f);
            }
            else
            {
                C.Accumulate(A);
            }
        }

        public static void CollapseColumnsAndAccumulate(this Matrix<double> C, Matrix<double> A, Matrix<double> B)
        {
            if (A.ColumnCount > 1)
            {
                C.Accumulate(A, B, 1.0f);
            }
            else
            {
                C.Accumulate(A);
            }
        }

        /// <summary>
        ///     this = alpha*A + this
        /// </summary>
        public static void Accumulate(this Matrix<float> x, Matrix<float> A, float alpha = 1.0f)
        {
            if (A.ColumnCount == 1)
            {
                SumVec(A, x, alpha);
            }
            else
            {
                var aa = A.AsColumnMajorArray();
                var xa = x.AsColumnMajorArray();
                if (alpha != 1.0f)
                {
                    Control.LinearAlgebraProvider.ScaleArray(alpha, aa, aa);
                }
                
                Control.LinearAlgebraProvider.AddArrays(xa, aa, xa);
            }
        }

        /// <summary>
        ///     this = alpha*A + this
        /// </summary>
        public static void Accumulate(this Matrix<double> x, Matrix<double> A, float alpha = 1.0f)
        {
            if (A.ColumnCount == 1)
            {
                SumVec(A, x, alpha);
            }
            else
            {
                var aa = A.AsColumnMajorArray();
                var xa = x.AsColumnMajorArray();
                if (alpha != 1.0f)
                {
                    Control.LinearAlgebraProvider.ScaleArray(alpha, aa, aa);
                }

                Control.LinearAlgebraProvider.AddArrays(xa, aa, xa);
            }
        }

        public static List<Matrix<T>> SplitColumns<T>(this Matrix<T> matrix, int columnCount) where T : struct, IEquatable<T>, IFormattable
        {
            if (matrix.ColumnCount < columnCount || matrix.ColumnCount % columnCount != 0)
                throw new ArgumentOutOfRangeException("columnCount", "Column count must be greater or equal to input matrix column count and must divide it with no remainder.");

            int cnt = matrix.ColumnCount / columnCount;
            var result = new List<Matrix<T>>(cnt);
            for (int i = 0; i < cnt; i++)
            {
                result.Add(Matrix<T>.Build.DenseOfColumns(matrix.EnumerateColumns(columnCount * i, columnCount)));
            }

            return result;
        }

        public static void CopyToArray<T>(this Matrix<T> matrix, T[] dest, ref int idx) where T : struct, IEquatable<T>, IFormattable
        {
            var ma = matrix.AsColumnMajorArray();
            if (dest.Length - idx < ma.Length)
            {
                throw new InvalidOperationException("Not enough space in target array!");
            }

            Array.Copy(ma, 0, dest, idx, ma.Length);
            idx += ma.Length;
        }

        public static void CopyFromArray<T>(this Matrix<T> matrix, T[] src, ref int idx) where T : struct, IEquatable<T>, IFormattable
        {
            var ma = matrix.AsColumnMajorArray();
            if (src.Length - idx < ma.Length)
            {
                throw new InvalidOperationException("Source array doesn't have enough data!");
            }

            Array.Copy(src, idx, ma, 0, ma.Length);
            idx += ma.Length;
        }

        public static bool EqualsTo(this Matrix<float> matrix, Matrix<float> other)
        {
            if (ReferenceEquals(matrix, other))
            {
                return true;
            }

            var m1 = matrix.AsColumnMajorArray();
            var m2 = other.AsColumnMajorArray();

            if (ReferenceEquals(m1, m2))
            {
                return true;
            }

            if (m1.Length != m2.Length)
            {
                return false;
            }

            for (int i = 0; i < m1.Length; i++)
            {
                if (Math.Abs(m1[i] - m2[i]) > 1e-5f)
                {
                    return false;
                }
            }

            return true;
        }

        public static bool EqualsTo(this Matrix<double> matrix, Matrix<double> other)
        {
            if (ReferenceEquals(matrix, other))
            {
                return true;
            }

            var m1 = matrix.AsColumnMajorArray();
            var m2 = other.AsColumnMajorArray();

            if (ReferenceEquals(m1, m2))
            {
                return true;
            }

            if (m1.Length != m2.Length)
            {
                return false;
            }

            for (int i = 0; i < m1.Length; i++)
            {
                if (Math.Abs(m1[i] - m2[i]) > 1e-5f)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        ///     result=alpha*x + y;
        /// </summary>
        private static void SumVec(Matrix<float> x, Matrix<float> y, float alpha)
        {
            if (y.ColumnCount > 1 || x.ColumnCount > 1)
                throw new Exception("Vector BLAS function is called with matrix argument!");

            if (y.RowCount != x.RowCount)
                throw new Exception("Vector dimensions must agree!");

            var ya = y.AsColumnMajorArray();
            Control.LinearAlgebraProvider.AddVectorToScaledVector(ya, alpha, x.AsColumnMajorArray(), ya);
        }

        /// <summary>
        ///     result=alpha*x + y;
        /// </summary>
        private static void SumVec(Matrix<double> x, Matrix<double> y, double alpha)
        {
            if (y.ColumnCount > 1 || x.ColumnCount > 1)
                throw new Exception("Vector BLAS function is called with matrix argument!");

            if (y.RowCount != x.RowCount)
                throw new Exception("Vector dimensions must agree!");

            var ya = y.AsColumnMajorArray();
            Control.LinearAlgebraProvider.AddVectorToScaledVector(ya, alpha, x.AsColumnMajorArray(), ya);
        }
    }
}