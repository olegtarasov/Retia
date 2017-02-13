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
            MathProvider<T>.Instance.ClampMatrix(matrix, min, max);
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
        ///     C = AB + (useC ? 1 : 0)*C
        /// </summary>
        public static void Accumulate<T>(this Matrix<T> C, Matrix<T> A, Matrix<T> B, Transpose transposeA = Transpose.DontTranspose, Transpose transposeB = Transpose.DontTranspose, bool useC = true) where T : struct, IEquatable<T>, IFormattable
        {
            ((ILinearAlgebraProvider<T>)Control.LinearAlgebraProvider).MatrixMultiplyWithUpdate(transposeA, transposeB, Matrix<T>.One, A.AsColumnMajorArray(), A.RowCount, A.ColumnCount, B.AsColumnMajorArray(), B.RowCount, B.ColumnCount, useC ? Matrix<T>.One : Matrix<T>.Zero, C.AsColumnMajorArray());
        }

        ///// <summary>
        /////     C = alpha*AB + beta*C
        ///// </summary>
        //public static void Accumulate<T>(this Matrix<T> C, Matrix<T> A, Matrix<T> B, float beta = 0.0f, float alpha = 1.0f, Transpose transposeA = Transpose.DontTranspose, Transpose transposeB = Transpose.DontTranspose) where T : struct, IEquatable<T>, IFormattable
        //{
        //    if (typeof(T) == typeof(float))
        //    {
        //        Control.LinearAlgebraProvider.MatrixMultiplyWithUpdate(transposeA, transposeB, alpha, A.AsColumnMajorArray() as float[], A.RowCount, A.ColumnCount, B.AsColumnMajorArray() as float[], B.RowCount, B.ColumnCount, beta, C.AsColumnMajorArray() as float[]);
        //    }
        //    else
        //    {
        //        Control.LinearAlgebraProvider.MatrixMultiplyWithUpdate(transposeA, transposeB, (double)alpha, A.AsColumnMajorArray() as double[], A.RowCount, A.ColumnCount, B.AsColumnMajorArray() as double[], B.RowCount, B.ColumnCount, (double)beta, C.AsColumnMajorArray() as double[]);
        //    }
        //}

        public static void CollapseColumnsAndAccumulate<T>(this Matrix<T> C, Matrix<T> A, Matrix<T> B) where T : struct, IEquatable<T>, IFormattable
        {
            if (A.ColumnCount > 1)
            {
                C.Accumulate(A, B);
            }
            else
            {
                C.Accumulate(A);
            }
        }

        /// <summary>
        ///     this = alpha*A + this
        /// </summary>
        public static void Accumulate<T>(this Matrix<T> x, Matrix<T> A/*, float alpha = 1.0f*/) where T : struct, IEquatable<T>, IFormattable
        {
            if (A.ColumnCount == 1)
            {
                SumVec(A, x/*, alpha*/);
            }
            else
            {
                var aa = A.AsColumnMajorArray();
                var xa = x.AsColumnMajorArray();
                //if (alpha != 1.0f)
                //{
                //    if (typeof(T) == typeof(float))
                //        Control.LinearAlgebraProvider.ScaleArray(alpha, aa as float[], aa as float[]);
                //    else
                //        Control.LinearAlgebraProvider.ScaleArray(alpha, aa as double[], aa as double[]);
                //}
                
                ((ILinearAlgebraProvider<T>)Control.LinearAlgebraProvider).AddArrays(xa, aa, xa);
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

        public static bool EqualsTo<T>(this Matrix<T> matrix, Matrix<T> other) where T : struct, IEquatable<T>, IFormattable
        {
            return MathProvider<T>.Instance.MatricesEqual(matrix, other);
        }

        /// <summary>
        ///     result=x + y;
        /// </summary>
        private static void SumVec<T>(Matrix<T> x, Matrix<T> y/*, float alpha*/) where T : struct, IEquatable<T>, IFormattable
        {
            if (y.ColumnCount > 1 || x.ColumnCount > 1)
                throw new Exception("Vector BLAS function is called with matrix argument!");

            if (y.RowCount != x.RowCount)
                throw new Exception("Vector dimensions must agree!");

            var ya = y.AsColumnMajorArray();
            ((ILinearAlgebraProvider<T>)Control.LinearAlgebraProvider).AddVectorToScaledVector(ya, Matrix<T>.One, x.AsColumnMajorArray(), ya);
        }
    }
}