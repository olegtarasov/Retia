using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using Retia.Interop;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Tests.Plumbing;
using Xunit;

namespace Retia.Tests.Interop
{
    public class CpuTransferTests : TransferTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.CpuTesting.Instance;
    }

    public abstract class TransferTestsBase : GpuTestsBase
    {
        [Fact]
        public void CanTransferMatrix()
        {
            var local = MatrixFactory.RandomMatrix<float>(2, 3, 5.0f);
            var remote = local.CloneMatrix();

            MutateMatrix(local);

            using (var ptrs = new MatrixPointersBag<float>(true, remote))
            {
                Interface.TestMatrixTransfer(ptrs.Definitions[0]);
            }

            local.ShouldMatrixEqualWithinError(remote);
        }

        [Fact]
        public void CanTransferMatrixRowMajor()
        {
            var local = MatrixFactory.RandomMatrix<float>(2, 3, 5.0f);
            var remote = local.CloneMatrix();

            MutateMatrixRowMajor(local);

            using (var ptrs = new MatrixPointersBag<float>(true, true, remote))
            {
                Interface.TestMatrixTransferRowMajor(ptrs.Definitions[0]);
            }

            local.ShouldMatrixEqualWithinError(remote);
        }

        [Fact]
        public void CanTransferWeight()
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
                Interface.TestWeightTransfer(ptrs.Definitions[0]);
            }

            local.Weight.ShouldMatrixEqualWithinError(remote.Weight);
            local.Gradient.ShouldMatrixEqualWithinError(remote.Gradient);
            local.Cache1.ShouldMatrixEqualWithinError(remote.Cache1);
            local.Cache2.ShouldMatrixEqualWithinError(remote.Cache2);
            local.CacheM.ShouldMatrixEqualWithinError(remote.CacheM);
        }

        [Fact]
        public void CanTransferWeightRowMajor()
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
                Interface.TestWeightTransferRowMajor(ptrs.Definitions[0]);
            }

            local.Weight.ShouldMatrixEqualWithinError(remote.Weight);
            local.Gradient.ShouldMatrixEqualWithinError(remote.Gradient);
            local.Cache1.ShouldMatrixEqualWithinError(remote.Cache1);
            local.Cache2.ShouldMatrixEqualWithinError(remote.Cache2);
            local.CacheM.ShouldMatrixEqualWithinError(remote.CacheM);
        }

        private static void MutateMatrix(Matrix<float> matrix)
        {
            for (int col = 0; col < matrix.ColumnCount; col++)
            {
                for (int row = 0; row < matrix.RowCount; row++)
                {
                    matrix[row, col] += row - col;
                }
            }
        }

        private static void MutateMatrixRowMajor(Matrix<float> matrix)
        {
            var la = matrix.ToRowMajorArray();
            for (int row = 0; row < matrix.RowCount; row++)
            {
                for (int col = 0; col < matrix.ColumnCount; col++)
                {
                    la[row * matrix.ColumnCount + col] += row - col;
                }
            }
            DenseColumnMajorMatrixStorage<float>.OfRowMajorArray(matrix.RowCount, matrix.ColumnCount, la).CopyTo(matrix.Storage);
        }
    }
}