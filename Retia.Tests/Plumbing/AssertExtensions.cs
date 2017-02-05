using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Single;
using Xunit;
using XunitShould;

namespace Retia.Tests.Plumbing
{
    public static class AssertExtensions
    {
        public static void ShouldArrayEqual(this float[] array1, float[] array2)
        {
            if (array1 == null && array2 == null)
            {
                return;
            }

            array1.ShouldNotBeNull();
            array2.ShouldNotBeNull();

            for (int i = 0; i < array1.Length; i++)
            {
                Math.Abs(array1[i] - array2[i]).ShouldBeLessThan(1e-5f);
            }
        }

        public static void ShouldEqualWithinError(this float val, float expected, float error = 1e-5f)
        {
            Math.Abs(val - expected).ShouldBeLessThan(error);
        }

        public static void ShouldHaveSize(this Matrix matrix, int rows, int cols)
        {
            matrix.RowCount.ShouldEqual(rows);
            matrix.ColumnCount.ShouldEqual(cols);
        }
    }
}