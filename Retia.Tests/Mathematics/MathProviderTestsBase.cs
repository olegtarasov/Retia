using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;
using Retia.Tests.Plumbing;
using Xunit;

namespace Retia.Tests.Mathematics
{
    public abstract class MathProviderTestsBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected MathProviderBase<T> MathProvider => MathProvider<T>.Instance;

        private static IEnumerable<object[]> GetActivationFuncsTestData()
        {
            yield return new object[] { MatrixFactory.RandomMatrix<T>(1, 1, 1.0f) };
            yield return new object[] { MatrixFactory.RandomMatrix<T>(1, 2, 1.0f) };
            yield return new object[] { MatrixFactory.RandomMatrix<T>(2, 1, 1.0f) };
            yield return new object[] { MatrixFactory.RandomMatrix<T>(4, 4, 1.0f) };
            yield return new object[] { MatrixFactory.RandomMatrix<T>(5, 3, 1.0f) };
            yield return new object[] { MatrixFactory.RandomMatrix<T>(4, 10, 1.0f) };
            yield return new object[] { MatrixFactory.RandomMatrix<T>(15, 15, 1.0f) };
        }

        [Theory]
        [MemberData(nameof(GetActivationFuncsTestData))]
        public void CanApplySigmoid(Matrix<T> matrix)
        {
            var src = matrix.CloneMatrix();
            var clone = matrix.CloneMatrix();

            MathProvider.ApplySigmoid2(matrix, clone);

            var matArr = matrix.AsColumnMajorArray();
            var cloneArr = clone.AsColumnMajorArray();
            var sourceArr = src.AsColumnMajorArray();

            for (int i = 0; i < sourceArr.Length; i++)
            {
                cloneArr[i].ShouldEqualWithinError(Sigmoid(sourceArr[i]));
                matArr[i].ShouldEqualWithinError(Sigmoid(sourceArr[i]));
            }
        }

        [Theory]
        [MemberData(nameof(GetActivationFuncsTestData))]
        public void CanApplyTanh(Matrix<T> matrix)
        {
            TestMatrix(matrix, MathProvider.ApplyTanh, Tanh);
        }

        protected abstract T Sigmoid(T input);
        protected abstract T Tanh(T input);

        private void TestMatrix(Matrix<T> matrix, Action<Matrix<T>> testAction, Func<T, T> checkFunc)
        {
            var clone = matrix.CloneMatrix();
            testAction(clone);

            var cloneArr = clone.AsColumnMajorArray();
            var sourceArr = matrix.AsColumnMajorArray();

            for (int i = 0; i < sourceArr.Length; i++)
            {
                cloneArr[i].ShouldEqualWithinError(checkFunc(sourceArr[i]));
            }
        }
    }
}