#if !CPUONLY
using Retia.Interop;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Tests.Interop;
using Retia.Tests.Plumbing;
using Xunit;

namespace Retia.Tests.Gpu
{
    public class GpuTransferTests : TransferTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.GpuTesting.Instance;

        [Fact]
        public void CanTransferWeightThroughNeuroWeight()
        {
            var weight = MatrixFactory.RandomMatrix<float>(2, 3, 5.0f);
            var local = new NeuroWeight<float>(weight);
            var remote = local.Clone();

            MutateMatrix(local.Weight);
            MutateMatrix(local.Gradient);
            MutateMatrix(local.Cache1);
            MutateMatrix(local.Cache2);
            MutateMatrix(local.CacheM);

            using (var ptrs = new WeightDefinitionBag<float>(remote))
            {
                GpuInterface.Testing.TestComplexWeightTransfer(ptrs.Definitions[0]);
            }

            local.Weight.ShouldMatrixEqualWithinError(remote.Weight);
            local.Gradient.ShouldMatrixEqualWithinError(remote.Gradient);
            local.Cache1.ShouldMatrixEqualWithinError(remote.Cache1);
            local.Cache2.ShouldMatrixEqualWithinError(remote.Cache2);
            local.CacheM.ShouldMatrixEqualWithinError(remote.CacheM);
        }

        [Fact]
        public void CanTransferWeightThroughNeuroWeightRowMajor()
        {
            var weight = MatrixFactory.RandomMatrix<float>(2, 3, 5.0f);
            var local = new NeuroWeight<float>(weight);
            var remote = local.Clone();

            MutateMatrixRowMajor(local.Weight);
            MutateMatrixRowMajor(local.Gradient);
            MutateMatrixRowMajor(local.Cache1);
            MutateMatrixRowMajor(local.Cache2);
            MutateMatrixRowMajor(local.CacheM);

            using (var ptrs = new WeightDefinitionBag<float>(true, remote))
            {
                GpuInterface.Testing.TestComplexWeightTransferRowMajor(ptrs.Definitions[0]);
            }

            local.Weight.ShouldMatrixEqualWithinError(remote.Weight);
            local.Gradient.ShouldMatrixEqualWithinError(remote.Gradient);
            local.Cache1.ShouldMatrixEqualWithinError(remote.Cache1);
            local.Cache2.ShouldMatrixEqualWithinError(remote.Cache2);
            local.CacheM.ShouldMatrixEqualWithinError(remote.CacheM);
        }
    }
}

#endif