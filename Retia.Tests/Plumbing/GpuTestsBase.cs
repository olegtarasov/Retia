using Retia.Interop;
using Retia.Mathematics;

namespace Retia.Tests.Plumbing
{
    public abstract class GpuTestsBase
    {
        protected readonly MathProviderBase<float> MathProvider = MathProvider<float>.Instance;
        protected abstract GpuInterface.TestingBase Interface { get; }
    }
}