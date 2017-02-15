using System;

namespace Retia.Tests.Mathematics
{
    public class SingleMathProviderTests : MathProviderTestsBase<float>
    {
        protected override float Sigmoid(float input) => 1.0f / (1.0f + (float)Math.Exp(-input));
        protected override float Tanh(float input) => (float)Math.Tanh(input);
    }
}