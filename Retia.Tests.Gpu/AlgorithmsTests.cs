using Retia.Gpu;
using Retia.Mathematics;
using Retia.Tests.Mathematics;
using Retia.Tests.Plumbing;
using Xunit;
using XunitShould;

namespace Retia.Tests.Gpu
{
    public class GpuAlgorithmsTests : AlgorithmsTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.GpuTesting.Instance;
    }
}