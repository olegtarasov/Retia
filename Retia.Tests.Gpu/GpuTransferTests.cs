#if !CPUONLY
using Retia.Interop;
using Retia.Tests.Interop;

namespace Retia.Tests.Gpu
{
    public class GpuTransferTests : TransferTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.GpuTesting.Instance;
    }
}

#endif