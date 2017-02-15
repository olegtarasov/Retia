using System;

namespace Retia.Tests.Mathematics
{
    public class DoubleMathProviderTests : MathProviderTestsBase<double>
    {
        protected override double Sigmoid(double input) => 1.0f / (1.0f + Math.Exp(-input));
        protected override double Tanh(double input) => Math.Tanh(input);
    }
}