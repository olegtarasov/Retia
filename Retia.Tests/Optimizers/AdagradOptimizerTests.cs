using Retia.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace Retia.Tests.Optimizers
{
    public class AdagradOptimizerTests : OptimizerTestBase
    {
        public AdagradOptimizerTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override OptimizerBase<float> GetOptimizer() => new AdagradOptimizer<float>();
    }
}