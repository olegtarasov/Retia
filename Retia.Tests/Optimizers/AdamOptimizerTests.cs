using Retia.Optimizers;
using Xunit.Abstractions;

namespace Retia.Tests.Optimizers
{
    public class AdamOptimizerTests : OptimizerTestBase
    {
        public AdamOptimizerTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override OptimizerBase<float> GetOptimizer()
        {
            return new AdamOptimizer<float>();
        }
    }
}