using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;
using Retia.Tests.Plumbing;
using Xunit;
using XunitShould;

namespace Retia.Tests.Mathematics
{
    public class ActivationFuncsTests
    {
        private static IEnumerable<object[]> GetActivationFuncsTestData()
        {
            yield return new object[] { DenseMatrix.Build.Random(1, 1) };
            yield return new object[] { DenseMatrix.Build.Random(1, 2) };
            yield return new object[] { DenseMatrix.Build.Random(2, 1) };
            yield return new object[] { DenseMatrix.Build.Random(4, 4) };
            yield return new object[] { DenseMatrix.Build.Random(5, 3) };
            yield return new object[] { DenseMatrix.Build.Random(4, 10) };
            yield return new object[] { DenseMatrix.Build.Random(15, 15) };
        }

        [Theory]
        [MemberData(nameof(GetActivationFuncsTestData))]
        public void CanApplySigmoid(Matrix matrix)
        {
            var src = matrix.CloneMatrix();
            var clone = matrix.CloneMatrix();
            
            ActivationFuncs.ApplySigmoid2(matrix, clone);

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
        public void CanApplyTanh(Matrix matrix)
        {
            TestMatrix(matrix, ActivationFuncs.ApplyTanh, Tanh);
        }

        private void TestMatrix(Matrix matrix, Action<Matrix> testAction, Func<float, float> checkFunc)
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

        private static float Sigmoid(float x) => 1.0f / (1.0f + (float)Math.Exp(-x));
        private static float Tanh(float x) => (float)Math.Tanh(x);
    }
}