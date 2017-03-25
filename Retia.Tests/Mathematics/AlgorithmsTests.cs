using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Gpu;
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
    public abstract class AlgorithmsTestsBase
    {
        private readonly MathProviderBase<float> MathProvider = MathProvider<float>.Instance;

        protected abstract GpuInterface.TestingBase Interface { get; }

        [Fact]
        public void CrossEntropyErrorsAreEqual()
        {
            var m1 = MatrixFactory.RandomMatrix<float>(100, 100, 1e-5f, 1.0f);
            var m2 = MatrixFactory.RandomMatrix<float>(100, 100, 1e-5f, 1.0f);

            using (var matrixPtrs = new HostMatrixPointers<float>(m1.CloneMatrix(), m2.CloneMatrix()))
            {
                double local = MathProvider.CrossEntropyError(m1, m2);
                double remote = Interface.TestCrossEntropyError(matrixPtrs.Definitions[0], matrixPtrs.Definitions[1]);

                double.IsNaN(local).ShouldBeFalse();
                double.IsNaN(remote).ShouldBeFalse();

                remote.ShouldEqualWithinError(local);
            }
        }

        [Fact]
        public void CrossEntropyBackpropagationsAreEqual()
        {
            var m1 = MatrixFactory.RandomMatrix<float>(100, 100, 1e-5f, 1.0f);
            var m2 = MatrixFactory.RandomMatrix<float>(100, 100, 1e-5f, 1.0f);
            var remoteResult = MatrixFactory.Create<float>(100, 100);

            Matrix<float> local;
            using (var matrixPtrs = new HostMatrixPointers<float>(m1.CloneMatrix(), m2.CloneMatrix(), remoteResult))
            {
                local = MathProvider.BackPropagateCrossEntropyError(m1, m2);
                Interface.TestCrossEntropyBackprop(matrixPtrs.Definitions[0], matrixPtrs.Definitions[1], matrixPtrs.Definitions[2]);
            }

            remoteResult.ShouldMatrixEqualWithinError(local);
        }

        private static IEnumerable<object[]> GetRMSPropTestData()
        {
            yield return new object[] { 5e-4f, 0.0f, 0.0f, 0.0f };
            yield return new object[] { 5e-4f, 0.5f, 0.0f, 0.0f };
            yield return new object[] { 5e-4f, 0.0f, 0.5f, 0.0f };
            yield return new object[] { 5e-4f, 0.0f, 0.0f, 0.5f };
            yield return new object[] { 5e-4f, 0.5f, 0.5f, 0.0f };
            yield return new object[] { 5e-4f, 0.0f, 0.5f, 0.5f };
            yield return new object[] { 5e-4f, 0.5f, 0.5f, 0.5f };
        }

        [Theory]
        [MemberData(nameof(GetRMSPropTestData))]
        public void RMSPropValuesAreEqual(float learningRate, float decayRate, float weightDecay, float momentum)
        {
            var local = new NeuroWeight<float>(MatrixFactory.RandomMatrix<float>(20, 20, 1e-2f));
            var remote = local.Clone();

            for (int i = 0; i < 100; i++)
            {
                var grad = MatrixFactory.RandomMatrix<float>(20, 20, 1.0f);

                grad.CopyTo(local.Gradient);
                grad.CopyTo(remote.Gradient);

                MathProvider.GravesRmsPropUpdate(weightDecay, learningRate, decayRate, momentum, local);

                using (var ptrs = new HostMatrixPointers<float>(remote.Weight, remote.Gradient, remote.Cache1, remote.Cache2, remote.CacheM))
                {
                    Interface.TestRMSPropUpdate(ptrs.Definitions[0], ptrs.Definitions[1], ptrs.Definitions[2], ptrs.Definitions[3], ptrs.Definitions[4],
                        learningRate, decayRate, momentum, weightDecay);
                }

                local.Weight.ShouldMatrixEqualWithinError(remote.Weight);
                local.Cache1.ShouldMatrixEqualWithinError(remote.Cache1);
                local.Cache2.ShouldMatrixEqualWithinError(remote.Cache2);
                local.CacheM.ShouldMatrixEqualWithinError(remote.CacheM);
                local.Gradient.ShouldMatrixEqualWithinError(remote.Gradient);
            }
        }

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

    public class CpuAlgorithmsTests : AlgorithmsTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.CpuTesting.Instance;
    }
}