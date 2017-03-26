using System;
using Retia.Interop;
using Retia.Mathematics;
using Xunit;
using XunitShould;

namespace Retia.Tests.Interop
{
    public class MatrixPointersTests
    {
        [Fact]
        public unsafe void CanPinMatrixPointers()
        {
            var m1 = MatrixFactory.Create<float>(2, 3);
            var m2 = MatrixFactory.Create<float>(2, 3);

            var ptrs = new MatrixPointersBag<float>(m1, m2);
            var p1 = ptrs[0];
            var p2 = ptrs[1];

            GC.Collect();

            ptrs[0].ShouldEqual(p1);
            ptrs[1].ShouldEqual(p2);

            fixed (void* fp1 = &m1.AsColumnMajorArray()[0], fp2 = &m2.AsColumnMajorArray()[0])
            {
                (p1.ToPointer() == fp1).ShouldBeTrue();
                (p2.ToPointer() == fp2).ShouldBeTrue();
            }

            ptrs.Dispose();
        }

        [Fact]
        public unsafe void CantAccesMatrixPointersWhenDisposed()
        {
            var m1 = MatrixFactory.Create<float>(2, 3);
            var ptrs = new MatrixPointersBag<float>(m1);

            fixed (void* fp1 = &m1.AsColumnMajorArray()[0])
            {
                (ptrs[0].ToPointer() == fp1).ShouldBeTrue();
            }

            ptrs.Dispose();

            Trap.Exception(() => ptrs[0]).ShouldBeInstanceOf<ObjectDisposedException>();
        }
    }
}