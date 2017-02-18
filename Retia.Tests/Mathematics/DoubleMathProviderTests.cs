using System;
using Xunit.Abstractions;

namespace Retia.Tests.Mathematics
{
    public class DoubleMathProviderTests : MathProviderTestsBase<double>
    {
        public DoubleMathProviderTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override double Sigmoid(double input) => 1.0f / (1.0f + Math.Exp(-input));
        protected override double Tanh(double input) => Math.Tanh(input);
    }
}