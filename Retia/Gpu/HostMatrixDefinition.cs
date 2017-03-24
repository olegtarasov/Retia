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

    [StructLayout(LayoutKind.Sequential)]
    public struct HostWeightDefinition
    {
        public int Rows;
        public int Columns;
        public int SeqLength;
        public IntPtr WeightPtr, GradPtr, Cache1Ptr, Cache2Ptr, CacheMPtr;
};
}