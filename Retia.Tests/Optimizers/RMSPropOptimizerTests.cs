using System;
using System.Data;
using MathNet.Numerics.LinearAlgebra;
using Retia.Neural;
using Retia.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace Retia.Tests.Optimizers
{
    public class RMSPropOptimizerTests : OptimizerTestBase
    {
        public RMSPropOptimizerTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override OptimizerBase<float> GetOptimizer() => new RMSPropOptimizer<float>(5e-4f, 0.99f, 0.0f, 0.0f);
    }
}