#if !CPUONLY
using System.Collections.Generic;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Tests.Plumbing;
using Xunit;
using Xunit.Sdk;
using XunitShould;

namespace Retia.Tests.Mathematics
{
    /// <summary>
    /// This suite tests .NET and GPU algorithms on same data to make sure they give
    /// identical results.
    /// </summary>
    public class GpuAlgorithmsTests
    {
        private readonly MathProviderBase<float> MathProvider = MathProvider<float>.Instance;

        //[Fact]
        //public void CrossEntropyErrorsAreEqual()
        //{
        //    var m1 = MatrixFactory.RandomMatrix<float>(10, 5, 1e-5f, 5.0f);
        //    var m2 = MatrixFactory.RandomMatrix<float>(10, 5, 1e-5f, 1.0f);

        //    double cpu = MathProvider.CrossEntropyError(m1, m2);
        //    double gpu = MathTester.TestCrossEntropyError(m1, m2);

        //    double.IsNaN(cpu).ShouldBeFalse();
        //    double.IsNaN(gpu).ShouldBeFalse();

        //    gpu.ShouldEqualWithinError(cpu);
        //}

        //[Fact]
        //public void CrossEntropyBackpropagationsAreEqual()
        //{
        //    var m1 = MatrixFactory.RandomMatrix<float>(10, 5, 1e-5f, 5.0f);
        //    var m2 = MatrixFactory.RandomMatrix<float>(10, 5, 1e-5f, 1.0f);
        //    var result = MatrixFactory.Create<float>(10, 5);

        //    var cpu = MathProvider.BackPropagateCrossEntropyError(m1, m2);
        //    MathTester.TestCrossEntropyBackpropagation(m1, m2, result);

        //    result.ShouldMatrixEqualWithinError(cpu);
        //}

        //private static IEnumerable<object[]> GetRMSPropTestData()
        //{
        //    yield return new object[] { 5e-4f, 0.0f, 0.0f, 0.0f };
        //    yield return new object[] { 5e-4f, 0.5f, 0.0f, 0.0f };
        //    yield return new object[] { 5e-4f, 0.0f, 0.5f, 0.0f };
        //    yield return new object[] { 5e-4f, 0.0f, 0.0f, 0.5f };
        //    yield return new object[] { 5e-4f, 0.5f, 0.5f, 0.0f };
        //    yield return new object[] { 5e-4f, 0.0f, 0.5f, 0.5f };
        //    yield return new object[] { 5e-4f, 0.5f, 0.5f, 0.5f };
        //}

        //[Theory]
        //[MemberData(nameof(GetRMSPropTestData))]
        //public void RMSPropValuesAreEqual(float learningRate, float decayRate, float weightDecay, float momentum)
        //{
        //    var hostWeight = new NeuroWeight<float>(MatrixFactory.Create<float>(10, 5));
        //    var gpuWeight = hostWeight.Clone();

        //    for (int i = 0; i < 100; i++)
        //    {
        //        var grad = MatrixFactory.RandomMatrix<float>(10, 5, 1.0f);

        //        grad.CopyTo(hostWeight.Gradient);
        //        grad.CopyTo(gpuWeight.Gradient);

        //        MathProvider.GravesRmsPropUpdate(weightDecay, learningRate, decayRate, momentum, hostWeight);
        //        MathTester.RMSPropOptimize(gpuWeight.Weight, gpuWeight.Gradient, gpuWeight.Cache1, gpuWeight.Cache2, gpuWeight.CacheM,
        //            learningRate, decayRate, momentum, weightDecay);

        //        hostWeight.Weight.ShouldMatrixEqualWithinError(gpuWeight.Weight);
        //        hostWeight.Cache1.ShouldMatrixEqualWithinError(gpuWeight.Cache1);
        //        hostWeight.Cache2.ShouldMatrixEqualWithinError(gpuWeight.Cache2);
        //        hostWeight.CacheM.ShouldMatrixEqualWithinError(gpuWeight.CacheM);
        //        hostWeight.Gradient.ShouldMatrixEqualWithinError(gpuWeight.Gradient);
        //    }
        //}

        //[Fact]
        //public void CanClampMatrix()
        //{
        //    var hostMatrix = MatrixFactory.ParseString<float>(@"2 5
        //                                           -6 1");
        //    var gpuMatrix = hostMatrix.Clone();


        //    hostMatrix.Clamp(-4.0f, 4.0f);
        //    hostMatrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(2.0f, -4.0f, 4.0f, 1.0f));

        //    MathTester.TestClampMatrix(gpuMatrix, 4.0f);
        //    gpuMatrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(2.0f, -4.0f, 4.0f, 1.0f));
        //}
    }
}

#endif