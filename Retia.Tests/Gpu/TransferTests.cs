using Retia.Gpu;
using Retia.Mathematics;
using Retia.Tests.Plumbing;
using Xunit;

namespace Retia.Tests.Gpu
{
    public abstract class TransferTestsBase : GpuTestsBase
    {
        [Fact]
        public void CanTransferMtrix()
        {
            var local = MatrixFactory.RandomMatrix<float>(2, 3, 5.0f);
            var remote = local.CloneMatrix();

            var la = local.AsColumnMajorArray();
            for (int col = 0; col < local.ColumnCount; col++)
            {
                for (int row = 0; row < local.RowCount; row++)
                {
                    local[row, col] += row - col;
                }
            }

            using (var ptrs = new HostMatrixPointers<float>(remote))
            {
                Interface.TestMatrixTransfer(ptrs.Definitions[0]);
            }

            local.ShouldMatrixEqualWithinError(remote);
        }
    }

    public class CpuTransferTests : TransferTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.CpuTesting.Instance;
    }
}