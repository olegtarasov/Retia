using System;
using System.Runtime.InteropServices;

namespace Retia.Gpu
{
    [StructLayout(LayoutKind.Sequential)]
    public struct HostMatrixDefinition
    {
        public int Rows;
        public int Columns;
        public int SeqLength;
        public IntPtr Pointer;
    }
}