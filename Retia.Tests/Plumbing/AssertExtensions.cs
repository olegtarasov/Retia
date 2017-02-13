using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Xunit;
using XunitShould;

namespace Retia.Tests.Plumbing
{
    public static class AssertExtensions
    {
        public static void ShouldArrayEqual<T>(this T[] array1, T[] array2) where T : struct, IEquatable<T>, IFormattable
        {
            if (array1 == null && array2 == null)
            {
                return;
            }

            array1.ShouldNotBeNull();
            array2.ShouldNotBeNull();

            for (int i = 0; i < array1.Length; i++)
            {
                array1[i].ShouldEqualWithinError(array2[i]);
            }
        }

        public static void ShouldEqualWithinError<T>(this T val, T expected, float error = 1e-5f) where T : struct, IEquatable<T>, IFormattable
        {
            if (typeof(T) == typeof(float))
                Math.Abs(Convert.ToSingle(val) - Convert.ToSingle(expected)).ShouldBeLessThan(error);
            else if (typeof(T) == typeof(double))
                Math.Abs(Convert.ToDouble(val) - Convert.ToDouble(expected)).ShouldBeLessThan(error);
            else
                throw new InvalidOperationException();
        }

        public static void ShouldHaveSize<T>(this Matrix<T> matrix, int rows, int cols) where T : struct, IEquatable<T>, IFormattable
        {
            matrix.RowCount.ShouldEqual(rows);
            matrix.ColumnCount.ShouldEqual(cols);
        }
    }
}