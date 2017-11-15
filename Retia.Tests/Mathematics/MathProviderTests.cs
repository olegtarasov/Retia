using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;
using Retia.Tests.Plumbing;
using Xunit;
using Xunit.Abstractions;

namespace Retia.Tests.Mathematics
{
    public class SingleMathProviderTests : MathProviderTestsBase<float>
    {
        public SingleMathProviderTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override float Sigmoid(float input) => 1.0f / (1.0f + (float)Math.Exp(-input));
        protected override float Tanh(float input) => (float)Math.Tanh(input);
    }

    public class DoubleMathProviderTests : MathProviderTestsBase<double>
    {
        public DoubleMathProviderTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override double Sigmoid(double input) => 1.0f / (1.0f + Math.Exp(-input));
        protected override double Tanh(double input) => Math.Tanh(input);
    }

    public abstract class MathProviderTestsBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected MathProviderTestsBase(ITestOutputHelper output)
        {
            _output = output;
        }

        private readonly ITestOutputHelper _output;

        #region Activation funcs

        protected MathProviderBase<T> MathProvider => MathProvider<T>.Instance;

        public static IEnumerable<object[]> GetActivationFuncsTestData()
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
        public void CanApplySingleSigmoid(Matrix<T> matrix)
        {
            var src = matrix.CloneMatrix();
            
            MathProvider.ApplySigmoid(matrix);

            var matArr = matrix.AsColumnMajorArray();
            var sourceArr = src.AsColumnMajorArray();

            for (int i = 0; i < sourceArr.Length; i++)
            {
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
            #endregion

        #region Error functions

        [Fact]
        public void CanCalculateCrossEntropyError()
        {
            var output = MatrixFactory.Create<T>(2, 2, 1.0f, 2.0f, 3.0f, 4.0f/*, 5.0f, 6.0f, 7.0f, 8.0f*/);
            var target = MatrixFactory.Create<T>(2, 2, 8.0f, 7.0f, 6.0f, 5.0f/*, 4.0f, 3.0f, 2.0f, 1.0f*/);

            _output.WriteLine($"Output\n{output.ToMatrixString()}");
            _output.WriteLine($"Target\n{target.ToMatrixString()}");

            double err = MathProvider.CrossEntropyError(output, target);
            double num = ((-(Math.Log(1.0f) * 8.0f) / 2) +
                          (-(Math.Log(2.0f) * 7.0f) / 2) +
                          (-(Math.Log(3.0f) * 6.0f) / 2) +
                          (-(Math.Log(4.0f) * 5.0f) / 2));

            err.ShouldEqualWithinError(num);

            _output.WriteLine($"Result: {err}");
        }

        #endregion

        #region Optimizer functions

        

        #endregion
    }
}