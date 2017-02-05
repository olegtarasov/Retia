using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Providers.LinearAlgebra;
using Retia.Mathematics;
using Retia.Tests.Plumbing;
using Xunit;
using XunitShould;

namespace Retia.Tests.Mathematics
{
    public partial class MatrixTests
    {
        #region String parsing

        [Fact]
        public void CanParseMatrixFromValidString()
        {
            var matrix = MatrixFactory.ParseString(@"1.0 2.0 3.0
                                                   4.0 5.0 6.0");

            matrix.ShouldHaveSize(2, 3);

            matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f });
        }

        [Fact]
        public void CanParseColumnVectorFromValidString()
        {
            var matrix = MatrixFactory.ParseString(@"1.0 
                                                   2.0
                                                   3.0");

            matrix.ShouldHaveSize(3, 1);

            matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 2.0f, 3.0f });
        }

        [Fact]
        public void CanParseColumnVectorFromValidStringWithTrailingNewlines()
        {
            var matrix = MatrixFactory.ParseString(@"1.0 
                                                   2.0
                                                   3.0

                                                     ");

            matrix.ShouldHaveSize(3, 1);

            matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 2.0f, 3.0f });
        }

        [Fact]
        public void CanParseRowVectorFromValidString()
        {
            var matrix = MatrixFactory.ParseString(@"1.0 2.0 3.0");

            matrix.ShouldHaveSize(1, 3);

            matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 2.0f, 3.0f });
        }

        [Fact]
        public void CantParseMatrixFromInsufficientData()
        {
            Trap.Exception(() => MatrixFactory.ParseString(@"1.0 2.0 3.0
                                                           4.0 5.0 
                                                           7.0 8.0 9.0"))
                .ShouldBeInstanceOf<InvalidOperationException>();
        }

        [Fact]
        public void CanParseMatrixWithCommaSeparators()
        {
            var matrix = MatrixFactory.ParseString(@"1,0 2,0 3,0
                                                   4,0 5,0 6,0");

            matrix.ShouldHaveSize(2, 3);

            matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f });
        }
        #endregion

        #region Accumulate operations

        private static IEnumerable<object[]> GetAccumulatePositiveTestData()
        {
            // Alpha-beta combinations
            yield return new object[] { Matrix5By3, Matrix3By6, DenseMatrix.Create(5, 6, 1), Transpose.DontTranspose, Transpose.DontTranspose, 1.0f, 0.0f };
            yield return new object[] { Matrix5By3, Matrix3By6, DenseMatrix.Create(5, 6, 1), Transpose.DontTranspose, Transpose.DontTranspose, 1.0f, 1.0f };
            yield return new object[] { Matrix5By3, Matrix3By6, DenseMatrix.Create(5, 6, 1), Transpose.DontTranspose, Transpose.DontTranspose, 0.0f, 1.0f };
            yield return new object[] { Matrix5By3, Matrix3By6, DenseMatrix.Create(5, 6, 1), Transpose.DontTranspose, Transpose.DontTranspose, 0.0f, 0.0f };

            // Transpose combinations
            yield return new object[] { Matrix3By6, Matrix3By6, DenseMatrix.Create(6, 6, 1), Transpose.Transpose, Transpose.DontTranspose, 1.0f, 1.0f };
            yield return new object[] { Matrix5By3, Matrix5By3, DenseMatrix.Create(5, 5, 1), Transpose.DontTranspose, Transpose.Transpose, 1.0f, 1.0f };
            yield return new object[] { Matrix3By6, Matrix5By3, DenseMatrix.Create(6, 5, 1), Transpose.Transpose, Transpose.Transpose, 1.0f, 1.0f };

            // Vector tests
            yield return new object[] { RowVector5, ColumnVector5, DenseMatrix.Create(1, 1, 1), Transpose.DontTranspose, Transpose.DontTranspose, 1.0f, 1.0f };
            yield return new object[] { ColumnVector5, RowVector5, DenseMatrix.Create(5, 5, 1), Transpose.DontTranspose, Transpose.DontTranspose, 1.0f, 1.0f };
            yield return new object[] { RowVector5, ColumnVector5, DenseMatrix.Create(5, 5, 1), Transpose.Transpose, Transpose.Transpose, 1.0f, 1.0f };

            // Matrix-vector tests
            yield return new object[] { Matrix5By3, ColumnVector3, DenseMatrix.Create(5, 1, 1), Transpose.DontTranspose, Transpose.DontTranspose, 1.0f, 1.0f };
            yield return new object[] { Matrix5By3, ColumnVector5, DenseMatrix.Create(3, 1, 1), Transpose.Transpose, Transpose.DontTranspose, 1.0f, 1.0f };
        }

        ///     C = beta * C + alpha * AB;
        [Theory]
        [MemberData(nameof(GetAccumulatePositiveTestData))]
        public void CanCalculateMatrixDotAndAccumulate(Matrix A, Matrix B, Matrix C, Transpose transposeA, Transpose transposeB, float alpha, float beta)
        {
            var tA = GetTestMatrix(A);
            var tB = GetTestMatrix(B);
            var tC = GetTestMatrix(C);

            C.Accumulate(A, B, beta, alpha, transposeA, transposeB);

            if (transposeA == Transpose.Transpose)
            {
                tA = (Matrix)tA.Transpose();
            }
            if (transposeB == Transpose.Transpose)
            {
                tB = (Matrix)tB.Transpose();
            }

            C.AsColumnMajorArray().ShouldArrayEqual((beta * tC + alpha * tA * tB).ToColumnMajorArray());
        }

        private static IEnumerable<object[]> GetAccumulateScalarPositiveTestData()
        {
            yield return new object[] { Matrix5By3, DenseMatrix.Create(5, 3, 1), 1.0f };
            yield return new object[] { Matrix5By3, DenseMatrix.Create(5, 3, 1), 0.0f };
            yield return new object[] { Matrix5By3, DenseMatrix.Create(5, 3, 1), 2.0f };

            yield return new object[] { ColumnVector5, DenseMatrix.Create(5, 1, 1), 1.0f };
            yield return new object[] { ColumnVector5, DenseMatrix.Create(5, 1, 1), 0.0f };
            yield return new object[] { ColumnVector5, DenseMatrix.Create(5, 1, 1), 2.0f };
        }

        ///     C = alpha*A + C
        [Theory]
        [MemberData(nameof(GetAccumulateScalarPositiveTestData))]
        public void CanMultiplyMatrixByScalarAndAccumulate(Matrix A, Matrix C, float alpha)
        {
            var tA = GetTestMatrix(A);
            var tC = GetTestMatrix(C);

            C.Accumulate(A, alpha);

            C.AsColumnMajorArray().ShouldArrayEqual((alpha * tA + tC).ToColumnMajorArray());
        }

        #endregion

        #region Column and row operations

        [Fact]
        public void CanTileColumnVector()
        {
            var matrix = MatrixFactory.ParseString(@"1
                                                   2
                                                   3
                                                   4");

            var result = matrix.TileColumns(3);
            result.ShouldHaveSize(4, 3);

            result.AsColumnMajorArray().ShouldArrayEqual(new float[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });
        }

        [Fact]
        public void CanTileMatrixColumns()
        {
            var result = Matrix3By6.TileColumns(3);

            result.ShouldHaveSize(3, 18);

            var test = DenseMatrix.OfColumns(Enumerable.Range(0, 3).SelectMany(x => Matrix3By6.EnumerateColumns()));

            result.AsColumnMajorArray().ShouldArrayEqual(test.AsColumnMajorArray());
        }

        [Fact]
        public void CanTileRowVector()
        {
            var matrix = MatrixFactory.ParseString(@"1 2 3 4");

            var result = matrix.TileRows(3);
            result.ShouldHaveSize(3, 4);

            result.AsColumnMajorArray().ShouldArrayEqual(new float[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 });
        }

        [Fact]
        public void CanTileMatrixRows()
        {
            var result = Matrix3By6.TileRows(3);

            result.ShouldHaveSize(9, 6);

            var test = DenseMatrix.OfRows(Enumerable.Range(0, 3).SelectMany(x => Matrix3By6.EnumerateRows()));

            result.AsColumnMajorArray().ShouldArrayEqual(test.AsColumnMajorArray());
        }

        #endregion

        #region Misc

        [Fact]
        public void CanSaveAndLoadMatrix()
        {
            var matrix = MatrixFactory.ParseString(@"1.0 2.0 3.0
                                                   4.0 5.0 6.0");

            matrix.ShouldHaveSize(2, 3);

            using (var file = new DisposableFile())
            {
                var read = file.WriteAndReadData(stream => matrix.Save(stream), MatrixFactory.Load);

                read.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f });
            }
        }

        [Fact]
        public void CanGenerateRandomMatrix()
        {
            using (new FakeRandom(1.0, 4.0, 2.0, 5.0, 3.0, 6.0))
            {
                var matrix = MatrixFactory.RandomMatrix(2, 3, 5);
                matrix.ShouldHaveSize(2, 3);
                matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f });
            }
        }

        [Fact]
        public void CanCreateRandomMaskMatrix()
        {
            using (new FakeRandom(0.4, 0.6, 0.5, 0.25, 0.0, 1.0))
            {
                var matrix = MatrixFactory.RandomMaskMatrix(2, 3, 0.5f);
                matrix.ShouldHaveSize(2, 3);
                matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f });
            }
        }

        [Fact]
        public void CanCompareMatrices()
        {
            MatrixFactory.ParseString(@"1.0 2.0 3.0
                                                    4.0 5.0 6.0
                                                    7.0 8.0 9.0").EqualsTo(
                MatrixFactory.ParseString(@"1.0 2.0 3.0
                                                    4.0 5.0 6.0
                                                    7.0 8.0 9.0")).ShouldBeTrue();

            Matrix3By6.EqualsTo(Matrix3By6_2).ShouldBeFalse();
            var self = Matrix3By6;
            self.EqualsTo(self).ShouldBeTrue();
            Matrix3By6.EqualsTo(Matrix5By3).ShouldBeFalse();

            var arr = new float[] {1, 2, 3, 4};
            var m1 = new DenseMatrix(2, 2, arr);
            var m2 = new DenseMatrix(2, 2, arr);
            m1.EqualsTo(m2).ShouldBeTrue();
        }

        [Fact]
        public void CanCloneMatrix()
        {
            var clone = Matrix3By6.CloneMatrix();

            clone.AsColumnMajorArray().ShouldArrayEqual(Matrix3By6.AsColumnMajorArray());

            clone[1, 1] = 42.0f;
            clone[1, 1].ShouldNotEqual(Matrix3By6[1, 1]);
        }

        [Fact]
        public void CanCopyMatrixToArray()
        {
            var dest = new float[Matrix5By3.RowCount * Matrix5By3.ColumnCount];
            int idx = 0;

            Matrix5By3.CopyToArray(dest, ref idx);

            Matrix5By3.AsColumnMajorArray().ShouldArrayEqual(dest);
            idx.ShouldEqual(Matrix5By3.RowCount * Matrix5By3.ColumnCount);
        }

        [Fact]
        public void CanRestoreMatrixFromArray()
        {
            var dest = new float[Matrix5By3.RowCount * Matrix5By3.ColumnCount];
            int idx = 0;

            Matrix5By3.CopyToArray(dest, ref idx);

            Matrix5By3.AsColumnMajorArray().ShouldArrayEqual(dest);
            idx.ShouldEqual(Matrix5By3.RowCount * Matrix5By3.ColumnCount);

            idx = 0;
            var mat = new DenseMatrix(5, 3);
            mat.CopyFromArray(dest, ref idx);

            mat.AsColumnMajorArray().ShouldArrayEqual(Matrix5By3.AsColumnMajorArray());
            idx.ShouldEqual(Matrix5By3.RowCount * Matrix5By3.ColumnCount);
        }

        [Fact]
        public void CanClampMatrix()
        {
            var matrix = MatrixFactory.ParseString(@"2 5
                                                   -6 1");

            matrix.Clamp(-4, 4);

            matrix.AsColumnMajorArray().ShouldArrayEqual(new[] { 2.0f, -4.0f, 4.0f, 1.0f });
        }

        [Fact]
        public void CanSplitMatrixColumns()
        {
            var result = Matrix3By6.SplitColumns(2);

            result.Count.ShouldEqual(3);
            for (int i = 0; i < result.Count; i++)
            {
                result[i].AsColumnMajorArray().ShouldArrayEqual(Matrix3By6.SubMatrix(0, Matrix3By6.RowCount, i * 2, 2).AsColumnMajorArray());
            }
        }

        [Fact]
        public void CanSplitMatrixColumnsWhenColumnCountEquals()
        {
            var result = Matrix3By6.SplitColumns(6);
            result.Count.ShouldEqual(1);
            result[0].AsColumnMajorArray().ShouldArrayEqual(Matrix3By6.AsColumnMajorArray());
        }

        [Fact]
        public void CantSplitMatrixColumnsWhenInvalidColumnCount()
        {
            Trap.Exception(() => Matrix3By6.SplitColumns(7)).ShouldBeInstanceOf<ArgumentOutOfRangeException>();
            Trap.Exception(() => Matrix3By6.SplitColumns(5)).ShouldBeInstanceOf<ArgumentOutOfRangeException>();
        }

        #endregion

        [Fact]
        public void CanPerformIndependentMul()
        {
            var m11 = MatrixFactory.RandomMatrix(3, 6, 5);
            var m2 = MatrixFactory.RandomMatrix(6, 8, 5);

            var res1 = m11 * m2;

            var m21 = MatrixFactory.RandomMatrix(3, 6, 5);

            var res2 = m21 * m2;

            var m1g = DenseMatrix.OfRows(m11.EnumerateRows().Concat(m21.EnumerateRows()));
            
            var resg = m1g * m2;

            var rg1 = resg.SubMatrix(0, 3, 0, 8);
            var rg2 = resg.SubMatrix(3, 3, 0, 8);

            rg1.AsColumnMajorArray().ShouldArrayEqual(res1.AsColumnMajorArray());
            rg2.AsColumnMajorArray().ShouldArrayEqual(res2.AsColumnMajorArray());
        }

        private Matrix GetTestMatrix(Matrix matrix)
        {
            return matrix.CloneMatrix();
        }
    }
}
