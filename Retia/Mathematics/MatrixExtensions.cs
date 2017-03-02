using System;
using System.Collections.Generic;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.Providers.LinearAlgebra;

namespace Retia.Mathematics
{
    /// <summary>
    /// Useful matrix operations that are not present in Math.NET.
    /// </summary>
    public static class MatrixExtensions
    {
        /// <summary>
        /// Returns the matrix underlying array length.
        /// </summary>
        public static int Length<T>(this Matrix<T> matrix) where T : struct, IEquatable<T>, IFormattable
        {
            return matrix.RowCount * matrix.ColumnCount;
        }

        /// <summary>
        /// Clones a matrix. New matrix storage is separated from the source matrix.
        /// </summary>
        public static Matrix<T> CloneMatrix<T>(this Matrix<T> matrix) where T : struct, IEquatable<T>, IFormattable
        {
            return Matrix<T>.Build.DenseOfMatrix(matrix);
        }

        /// <summary>
        /// Clips matrix values to the range of [min;max].
        /// </summary>
        /// <param name="matrix">Matrix to clamp.</param>
        /// <param name="min">Minimum value.</param>
        /// <param name="max">Maximum value.</param>
        public static void Clamp<T>(this Matrix<T> matrix, T min, T max) where T : struct, IEquatable<T>, IFormattable
        {
            MathProvider<T>.Instance.ClampMatrix(matrix, min, max);
        }

        /// <summary>
        /// Tiles a matrix horizontally <see cref="count" /> times and returns the result.
        /// </summary>
        /// <param name="matrix">Matrix to tile.</param>
        /// <param name="count">The number of times to tile.</param>
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

        /// <summary>
        /// Tiles a matrix vertically <see cref="count" /> times and returns the result.
        /// </summary>
        /// <param name="matrix">Matrix to tile.</param>
        /// <param name="count">The number of times to tile.</param>
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
        /// Subtracts <see cref="b"/> from <see cref="a"/> with result written to
        /// </summary>
        /// <param name="a">Minuend.</param>
        /// <param name="b">Subtrahend.</param>
        public static void SubtractInplace<T>(this Matrix<T> a, Matrix<T> b) where T : struct, IEquatable<T>, IFormattable
        {
            var aa = a.AsColumnMajorArray();
            var ba = b.AsColumnMajorArray();

            ((ILinearAlgebraProvider<T>)Control.LinearAlgebraProvider).SubtractArrays(aa, ba, aa);
        }

        /// <summary>
        /// Performs a BLAS operation:
        /// 
        ///     C = AB + (useC ? 1 : 0)*C
        /// </summary>
        public static void Accumulate<T>(this Matrix<T> C, Matrix<T> A, Matrix<T> B, Transpose transposeA = Transpose.DontTranspose, Transpose transposeB = Transpose.DontTranspose, bool useC = true) where T : struct, IEquatable<T>, IFormattable
        {
            ((ILinearAlgebraProvider<T>)Control.LinearAlgebraProvider).MatrixMultiplyWithUpdate(transposeA, transposeB, Matrix<T>.One, A.AsColumnMajorArray(), A.RowCount, A.ColumnCount, B.AsColumnMajorArray(), B.RowCount, B.ColumnCount, useC ? Matrix<T>.One : Matrix<T>.Zero, C.AsColumnMajorArray());
        }

        /// <summary>
        /// If A has more than one column, computes
        /// 
        ///     C = C + AB, where B is usually a singular column vector
        /// 
        /// If A has one column, computes
        /// 
        ///     C = C + A
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="C"></param>
        /// <param name="A"></param>
        /// <param name="B"></param>
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
        /// Computes
        /// 
        ///     C = A + C
        /// </summary>
        public static void Accumulate<T>(this Matrix<T> C, Matrix<T> A/*, float alpha = 1.0f*/) where T : struct, IEquatable<T>, IFormattable
        {
            if (A.ColumnCount == 1)
            {
                SumVec(A, C/*, alpha*/);
            }
            else
            {
                var aa = A.AsColumnMajorArray();
                var xa = C.AsColumnMajorArray();
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

        /// <summary>
        /// Splits a Nx(MxK) matrix into K NxM matrices.
        /// </summary>
        /// <param name="matrix">Nx(MxK) matrix</param>
        /// <param name="columnCount">M</param>
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

        /// <summary>
        /// Copies underlying matrix array to another array.
        /// </summary>
        /// <param name="matrix">Matrix to copy.</param>
        /// <param name="dest">Destination array.</param>
        /// <param name="idx">Destination start index which is increased after copying.</param>
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

        /// <summary>
        /// Copies an array into underlying matrix array.
        /// </summary>
        /// <param name="matrix">Matrix to copy to.</param>
        /// <param name="src">Source array.</param>
        /// <param name="idx">Source start index which is increased after copying.</param>
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

        /// <summary>
        /// Tests two matrices for equality with error margin of 10e-7.
        /// </summary>
        public static bool EqualsTo<T>(this Matrix<T> matrix, Matrix<T> other) where T : struct, IEquatable<T>, IFormattable
        {
            return MathProvider<T>.Instance.MatricesEqual(matrix, other);
        }

        /// <summary>
        ///     result=C + y;
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