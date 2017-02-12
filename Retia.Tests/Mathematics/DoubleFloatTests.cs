using System;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Xunit;
using XunitShould;

namespace Retia.Tests.Mathematics
{
    public class DoubleFloatTests
    {
        [Fact]
        public void Test()
        {
            Control.UseNativeMKL();

            var mf1 = Matrix<float>.Build.Random(10, 10);
            var mf2 = Matrix<float>.Build.Random(10, 10);
            var md1 = Matrix<double>.Build.Dense(10, 10, mf1.AsColumnMajorArray().Select(x => (double)x).ToArray());
            var md2 = Matrix<double>.Build.Dense(10, 10, mf2.AsColumnMajorArray().Select(x => (double)x).ToArray());

            var mfa = mf1.AsColumnMajorArray();
            var mda = md1.AsColumnMajorArray();

            for (int i = 0; i < mfa.Length; i++)
            {
                double d = mda[i] - mfa[i];
                Math.Abs(d).ShouldBeLessThan(1e-10);
            }

            var rf = mf1 * mf2;
            rf = rf * mf1;
            rf = rf * mf2;

            var rd = md1 * md2;
            rd = rd * md1;
            rd = rd * md2;

            var rfa = rf.AsColumnMajorArray();
            var rda = rd.AsColumnMajorArray();

            for (int i = 0; i < rfa.Length; i++)
            {
                double d = rda[i] - rfa[i];
                Math.Abs(d).ShouldBeLessThan(1e-4);
            }
        }
    }
}