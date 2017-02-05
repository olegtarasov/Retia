using System;
using System.Data;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Neural;
using Retia.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace Retia.Tests.Optimizers
{
    public class RMSPropOptimizerTests
    {
        private readonly ITestOutputHelper _output;

        public RMSPropOptimizerTests(ITestOutputHelper output)
        {
            _output = output;
        }

        //[Fact]
        public void CanOptimize()
        {
            var weight = new NeuroWeight(new DenseMatrix(2, 1));
            var optimizer = new RMSPropOptimizer(5e-4f, 0.99f, 0.0f, 0.0f);

            _output.WriteLine($"Rosenbrock: {Rosenbrock(weight.Weight)}");
            _output.WriteLine(weight.Weight.ToMatrixString());
            for (int i = 0; i < 10000; i++)
            {
                RosenbrockGrad(weight.Weight, weight.Gradient);
                optimizer.Optimize(weight);
                _output.WriteLine($"Rosenbrock: {Rosenbrock(weight.Weight)}");
                _output.WriteLine(weight.Weight.ToMatrixString());
            }
        }

        private float Rosenbrock(Matrix matrix)
        {
            var arr = matrix.AsColumnMajorArray();
            return (float)(100 * Math.Pow(arr[1] - Math.Pow(arr[0], 2), 2) + Math.Pow(1 - arr[0], 2));
        }

        private void RosenbrockGrad(Matrix matrix, Matrix grad)
        {
            var mArr = matrix.AsColumnMajorArray();
            var gArr = grad.AsColumnMajorArray();

            gArr[0] = (float)(-400 * (mArr[1] - Math.Pow(mArr[0], 2)) * mArr[0] - 2 * (1 - mArr[0]));
            gArr[1] = (float)(200 * (mArr[1] - Math.Pow(mArr[0], 2)));
        }
    }
}