#if !CPUONLY
using Retia.Gpu;

namespace Retia.Tests.Gpu
{
    public class GpuTransferTests : TransferTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.GpuTesting.Instance;
    }
}

#endif