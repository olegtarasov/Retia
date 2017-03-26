#if !CPUONLY
using Retia.Interop;
using Retia.Tests.Mathematics;

namespace Retia.Tests.Gpu
{
    public class GpuAlgorithmsTests : AlgorithmsTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.GpuTesting.Instance;
    }
}
#endif