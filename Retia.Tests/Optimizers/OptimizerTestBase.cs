using System;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Retia.Neural;
using Retia.Optimizers;
using Retia.Tests.Plumbing;
using Xunit;
using Xunit.Abstractions;
using XunitShould;

namespace Retia.Tests.Optimizers
{
    public abstract class OptimizerTestBase
    {
        private readonly ITestOutputHelper _output;

        public OptimizerTestBase(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void CanOptimize()
        {
            CanOptimizeRosenbrock(GetOptimizer());
        }

        protected void CanOptimizeRosenbrock(OptimizerBase<float> optimizer)
        {
            var weight = new NeuroWeight<float>(Matrix<float>.Build.Dense(2, 1));
            
            //_output.WriteLine($"Rosenbrock: {Rosenbrock(weight.Weight)}");
            //_output.WriteLine(weight.Weight.ToMatrixString());

            var watch = new Stopwatch();
            watch.Start();
            for (int i = 0; i < 10000; i++)
            {
                RosenbrockGrad(weight.Weight, weight.Gradient);
                optimizer.Optimize(weight);
                //_output.WriteLine($"Rosenbrock: {Rosenbrock(weight.Weight)}");
                //_output.WriteLine(weight.Weight.ToMatrixString());
            }
            watch.Stop();

            double result = Rosenbrock(weight.Weight);
            result.ShouldBeLessThanOrEqualTo(6e-5);

            _output.WriteLine($"Rosenbrock: {result}");
            _output.WriteLine($"Optimized in {watch.Elapsed}");
        }

        protected float Rosenbrock(Matrix<float> matrix)
        {
            var arr = matrix.AsColumnMajorArray();
            return (float)(100 * Math.Pow(arr[1] - Math.Pow(arr[0], 2), 2) + Math.Pow(1 - arr[0], 2));
        }

        protected void RosenbrockGrad(Matrix<float> matrix, Matrix<float> grad)
        {
            var mArr = matrix.AsColumnMajorArray();
            var gArr = grad.AsColumnMajorArray();

            gArr[0] = (float)(-400 * (mArr[1] - Math.Pow(mArr[0], 2)) * mArr[0] - 2 * (1 - mArr[0]));
            gArr[1] = (float)(200 * (mArr[1] - Math.Pow(mArr[0], 2)));
        }

        protected abstract OptimizerBase<float> GetOptimizer();
    }
}