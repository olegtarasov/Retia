using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using Retia.Mathematics;
using Retia.Tests.Plumbing;
using Xunit;
using XunitShould;

namespace Retia.Tests.Mathematics
{
    public class DoubleMatrixTests : MatrixTestsBase<double>
    {
    }

    public class SingleMatrixTests : MatrixTestsBase<float>
    {
    }

    public abstract partial class MatrixTestsBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected MathProviderBase<T> MathProvider => MathProvider<T>.Instance;

        #region String parsing

        [Fact]
        public void CanParseMatrixFromValidString()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1.0 2.0 3.0
                                                   4.0 5.0 6.0");

            matrix.ShouldHaveSize(2, 3);

            matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f));
        }

        [Fact]
        public void CanParseColumnVectorFromValidString()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1.0 
                                                   2.0
                                                   3.0");

            matrix.ShouldHaveSize(3, 1);

            matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 2.0f, 3.0f));
        }

        [Fact]
        public void CanParseColumnVectorFromValidStringWithTrailingNewlines()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1.0 
                                                   2.0
                                                   3.0

                                                     ");

            matrix.ShouldHaveSize(3, 1);

            matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 2.0f, 3.0f));
        }

        [Fact]
        public void CanParseRowVectorFromValidString()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1.0 2.0 3.0");

            matrix.ShouldHaveSize(1, 3);

            matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 2.0f, 3.0f));
        }

        [Fact]
        public void CantParseMatrixFromInsufficientData()
        {
            Trap.Exception(() => MatrixFactory.ParseString<T>(@"1.0 2.0 3.0
                                                           4.0 5.0 
                                                           7.0 8.0 9.0"))
                .ShouldBeInstanceOf<InvalidOperationException>();
        }

        [Fact]
        public void CanParseMatrixWithCommaSeparators()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1,0 2,0 3,0
                                                   4,0 5,0 6,0");

            matrix.ShouldHaveSize(2, 3);

            matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f));
        }
        #endregion

        #region Accumulate operations

        public static IEnumerable<object[]> GetAccumulatePositiveTestData()
        {
            // Alpha-beta combinations
            yield return new object[] { Matrix5By3, Matrix3By6, Matrix<T>.Build.Dense(5, 6, Matrix<T>.One), Transpose.DontTranspose, Transpose.DontTranspose, false };
            yield return new object[] { Matrix5By3, Matrix3By6, Matrix<T>.Build.Dense(5, 6, Matrix<T>.One), Transpose.DontTranspose, Transpose.DontTranspose, true };
            
            // Transpose combinations
            yield return new object[] { Matrix3By6, Matrix3By6, Matrix<T>.Build.Dense(6, 6, Matrix<T>.One), Transpose.Transpose, Transpose.DontTranspose, true };
            yield return new object[] { Matrix5By3, Matrix5By3, Matrix<T>.Build.Dense(5, 5, Matrix<T>.One), Transpose.DontTranspose, Transpose.Transpose, true };
            yield return new object[] { Matrix3By6, Matrix5By3, Matrix<T>.Build.Dense(6, 5, Matrix<T>.One), Transpose.Transpose, Transpose.Transpose, true };

            // Vector tests
            yield return new object[] { RowVector5, ColumnVector5, Matrix<T>.Build.Dense(1, 1, Matrix<T>.One), Transpose.DontTranspose, Transpose.DontTranspose, true};
            yield return new object[] { ColumnVector5, RowVector5, Matrix<T>.Build.Dense(5, 5, Matrix<T>.One), Transpose.DontTranspose, Transpose.DontTranspose, true};
            yield return new object[] { RowVector5, ColumnVector5, Matrix<T>.Build.Dense(5, 5, Matrix<T>.One), Transpose.Transpose, Transpose.Transpose, true};

            // Matrix-vector tests
            yield return new object[] { Matrix5By3, ColumnVector3, Matrix<T>.Build.Dense(5, 1, Matrix<T>.One), Transpose.DontTranspose, Transpose.DontTranspose, true};
            yield return new object[] { Matrix5By3, ColumnVector5, Matrix<T>.Build.Dense(3, 1, Matrix<T>.One), Transpose.Transpose, Transpose.DontTranspose, true};
        }

        ///     C = beta * C + alpha * AB;
        [Theory]
        [MemberData(nameof(GetAccumulatePositiveTestData))]
        public void CanCalculateMatrixDotAndAccumulate(Matrix<T> A, Matrix<T> B, Matrix<T> C, Transpose transposeA, Transpose transposeB, bool useC)
        {
            var tA = GetTestMatrix(A);
            var tB = GetTestMatrix(B);
            var tC = GetTestMatrix(C);

            C.Accumulate(A, B, transposeA, transposeB, useC);

            if (transposeA == Transpose.Transpose)
            {
                tA = tA.Transpose();
            }
            if (transposeB == Transpose.Transpose)
            {
                tB = tB.Transpose();
            }

            C.AsColumnMajorArray().ShouldArrayEqualWithinError(((useC ? Matrix<T>.One : Matrix<T>.Zero) * tC + tA * tB).ToColumnMajorArray());
        }

        public static IEnumerable<object[]> GetAccumulateScalarPositiveTestData()
        {
            yield return new object[] { Matrix5By3, Matrix<T>.Build.Dense(5, 3, Matrix<T>.One), 1.0f };
            yield return new object[] { Matrix5By3, Matrix<T>.Build.Dense(5, 3, Matrix<T>.One), 0.0f };
            yield return new object[] { Matrix5By3, Matrix<T>.Build.Dense(5, 3, Matrix<T>.One), 2.0f };

            yield return new object[] { ColumnVector5, Matrix<T>.Build.Dense(5, 1, Matrix<T>.One), 1.0f };
            yield return new object[] { ColumnVector5, Matrix<T>.Build.Dense(5, 1, Matrix<T>.One), 0.0f };
            yield return new object[] { ColumnVector5, Matrix<T>.Build.Dense(5, 1, Matrix<T>.One), 2.0f };
        }

        ///     C = alpha*A + C
        [Theory]
        [MemberData(nameof(GetAccumulateScalarPositiveTestData))]
        public void CanAccumulateMatrices(Matrix<T> A, Matrix<T> C, float alpha)
        {
            var tA = GetTestMatrix(A);
            var tC = GetTestMatrix(C);

            C.Accumulate(A);

            C.AsColumnMajorArray().ShouldArrayEqualWithinError((tA + tC).ToColumnMajorArray());
        }

        #endregion

        #region Column and row operations

        [Fact]
        public void CanTileColumnVector()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1
                                                   2
                                                   3
                                                   4");

            var result = matrix.TileColumns(3);
            result.ShouldHaveSize(4, 3);

            result.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4));
        }

        [Fact]
        public void CanTileMatrixColumns()
        {
            var result = Matrix3By6.TileColumns(3);

            result.ShouldHaveSize(3, 18);

            var test = Matrix<T>.Build.DenseOfColumns(Enumerable.Range(0, 3).SelectMany(x => Matrix3By6.EnumerateColumns()));

            result.AsColumnMajorArray().ShouldArrayEqualWithinError(test.AsColumnMajorArray());
        }

        [Fact]
        public void CanTileRowVector()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1 2 3 4");

            var result = matrix.TileRows(3);
            result.ShouldHaveSize(3, 4);

            result.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4));
        }

        [Fact]
        public void CanTileMatrixRows()
        {
            var result = Matrix3By6.TileRows(3);

            result.ShouldHaveSize(9, 6);

            var test = Matrix<T>.Build.DenseOfRows(Enumerable.Range(0, 3).SelectMany(x => Matrix3By6.EnumerateRows()));

            result.AsColumnMajorArray().ShouldArrayEqualWithinError(test.AsColumnMajorArray());
        }

        #endregion

        #region Misc

        [Fact]
        public void CanSaveAndLoadMatrix()
        {
            var matrix = MatrixFactory.ParseString<T>(@"1.0 2.0 3.0
                                                   4.0 5.0 6.0");

            matrix.ShouldHaveSize(2, 3);

            using (var file = new DisposableFile())
            {
                var read = file.WriteAndReadData(stream => matrix.Save(stream), MatrixFactory.Load<T>);

                read.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f));
            }
        }

        [Fact]
        public void CanGenerateRandomMatrix()
        {
            using (new FakeRandom(1.0, 4.0, 2.0, 5.0, 3.0, 6.0))
            {
                var matrix = MatrixFactory.RandomMatrix<T>(2, 3, 5);
                matrix.ShouldHaveSize(2, 3);
                matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f));
            }
        }

        [Fact]
        public void CanCreateRandomMaskMatrix()
        {
            using (new FakeRandom(0.4, 0.6, 0.5, 0.25, 0.0, 1.0))
            {
                var matrix = MatrixFactory.RandomMaskMatrix<T>(2, 3, 0.5f);
                matrix.ShouldHaveSize(2, 3);
                matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f));
            }
        }

        [Fact]
        public void CanCompareMatrices()
        {
            MatrixFactory.ParseString<T>(@"1.0 2.0 3.0
                                                    4.0 5.0 6.0
                                                    7.0 8.0 9.0").EqualsTo(
                MatrixFactory.ParseString<T>(@"1.0 2.0 3.0
                                                    4.0 5.0 6.0
                                                    7.0 8.0 9.0")).ShouldBeTrue();

            Matrix3By6.EqualsTo(Matrix3By6_2).ShouldBeFalse();
            var self = Matrix3By6;
            self.EqualsTo(self).ShouldBeTrue();
            Matrix3By6.EqualsTo(Matrix5By3).ShouldBeFalse();

            var arr = MathProvider.Array(1, 2, 3, 4);
            var m1 = Matrix<T>.Build.Dense(2, 2, arr);
            var m2 = Matrix<T>.Build.Dense(2, 2, arr);
            m1.EqualsTo(m2).ShouldBeTrue();
        }

        [Fact]
        public void CanCloneMatrix()
        {
            var clone = Matrix3By6.CloneMatrix();

            clone.AsColumnMajorArray().ShouldArrayEqualWithinError(Matrix3By6.AsColumnMajorArray());

            clone[1, 1] = MathProvider.Scalar(42.0f);
            clone[1, 1].ShouldNotEqual(Matrix3By6[1, 1]);
        }

        [Fact]
        public void CanCopyMatrixToArray()
        {
            var dest = new T[Matrix5By3.RowCount * Matrix5By3.ColumnCount];
            int idx = 0;

            Matrix5By3.CopyToArray(dest, ref idx);

            Matrix5By3.AsColumnMajorArray().ShouldArrayEqualWithinError(dest);
            idx.ShouldEqual(Matrix5By3.RowCount * Matrix5By3.ColumnCount);
        }

        [Fact]
        public void CanRestoreMatrixFromArray()
        {
            var dest = new T[Matrix5By3.RowCount * Matrix5By3.ColumnCount];
            int idx = 0;

            Matrix5By3.CopyToArray(dest, ref idx);

            Matrix5By3.AsColumnMajorArray().ShouldArrayEqualWithinError(dest);
            idx.ShouldEqual(Matrix5By3.RowCount * Matrix5By3.ColumnCount);

            idx = 0;
            var mat = Matrix<T>.Build.Dense(5, 3);
            mat.CopyFromArray(dest, ref idx);

            mat.AsColumnMajorArray().ShouldArrayEqualWithinError(Matrix5By3.AsColumnMajorArray());
            idx.ShouldEqual(Matrix5By3.RowCount * Matrix5By3.ColumnCount);
        }

        [Fact]
        public void CanClampMatrix()
        {
            var matrix = MatrixFactory.ParseString<T>(@"2 5
                                                   -6 1");

            matrix.Clamp(MathProvider.Scalar(-4), MathProvider.Scalar(4));

            matrix.AsColumnMajorArray().ShouldArrayEqualWithinError(MathProvider.Array(2.0f, -4.0f, 4.0f, 1.0f));
        }

        [Fact]
        public void CanSplitMatrixColumns()
        {
            var result = Matrix3By6.SplitColumns(2);

            result.Count.ShouldEqual(3);
            for (int i = 0; i < result.Count; i++)
            {
                result[i].AsColumnMajorArray().ShouldArrayEqualWithinError(Matrix3By6.SubMatrix(0, Matrix3By6.RowCount, i * 2, 2).AsColumnMajorArray());
            }
        }

        [Fact]
        public void CanSplitMatrixColumnsWhenColumnCountEquals()
        {
            var result = Matrix3By6.SplitColumns(6);
            result.Count.ShouldEqual(1);
            result[0].AsColumnMajorArray().ShouldArrayEqualWithinError(Matrix3By6.AsColumnMajorArray());
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
            var m11 = MatrixFactory.RandomMatrix<T>(3, 6, 5);
            var m2 = MatrixFactory.RandomMatrix<T>(6, 8, 5);

            var res1 = m11 * m2;

            var m21 = MatrixFactory.RandomMatrix<T>(3, 6, 5);

            var res2 = m21 * m2;

            var m1g = Matrix<T>.Build.DenseOfRows(m11.EnumerateRows().Concat(m21.EnumerateRows()));
            
            var resg = m1g * m2;

            var rg1 = resg.SubMatrix(0, 3, 0, 8);
            var rg2 = resg.SubMatrix(3, 3, 0, 8);

            rg1.AsColumnMajorArray().ShouldArrayEqualWithinError(res1.AsColumnMajorArray());
            rg2.AsColumnMajorArray().ShouldArrayEqualWithinError(res2.AsColumnMajorArray());
        }

        private Matrix<T> GetTestMatrix(Matrix<T> matrix)
        {
            return matrix.CloneMatrix();
        }
    }
}
