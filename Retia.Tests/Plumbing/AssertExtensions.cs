using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using Xunit;
using XunitShould;

namespace Retia.Tests.Plumbing
{
    public static class AssertExtensions
    {
        public static void ShouldArrayEqual(this double[] array1, double[] array2)
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

        public static void ShouldEqualWithinError(this double val, double expected, double error = 1e-5f)
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