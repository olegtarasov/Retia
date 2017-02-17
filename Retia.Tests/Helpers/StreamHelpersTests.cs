using System.IO;
using Retia.Helpers;
using Retia.Tests.Plumbing;
using Xunit;
using XunitShould;

namespace Retia.Tests.Helers
{
    public class StreamHelpersTests
    {
        [Fact]
        public void CanCreateNonGreedyReader()
        {
            var arr = new byte[] {0xDE, 0xAD, 0xBE, 0xEF};
            using (var ms = new MemoryStream(arr))
            {
                ms.ReadByte().ShouldEqual(0xDE);
                using (var reader = ms.NonGreedyReader())
                {
                    reader.ReadByte().ShouldEqual((byte)0xAD);
                }
                ms.ReadByte().ShouldEqual(0xBE);
            }
        }

        [Fact]
        public void CanCreateNonGreedyWriter()
        {
            using (var ms = new MemoryStream())
            {
                ms.WriteByte(0xDE);
                using (var writer = ms.NonGreedyWriter())
                {
                    writer.Write((byte)0xAD);
                }
                ms.WriteByte(0xBE);
                ms.WriteByte(0xEF);

                var arr = ms.ToArray();
                arr.Length.ShouldEqual(4);
                arr.ShouldEnumerateEqual((byte)0xDE, (byte)0xAD, (byte)0xBE, (byte)0xEF);
            }
        }
    }
}