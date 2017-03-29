using System;
using System.Runtime.InteropServices;

namespace Retia.Interop
{
    [StructLayout(LayoutKind.Sequential)]
    public struct MatrixDefinition
    {
        public MatrixDefinition(int rows, int columns, int seqLength, IntPtr pointer)
        {
            Rows = rows;
            Columns = columns;
            SeqLength = seqLength;
            Pointer = pointer;
        }

        public int Rows;
        public int Columns;
        public int SeqLength;
        public IntPtr Pointer;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct WeightDefinition
    {
        public WeightDefinition(int rows, int columns, int seqLength, IntPtr weightPtr, IntPtr gradPtr, IntPtr cache1Ptr, IntPtr cache2Ptr, IntPtr cacheMPtr)
        {
            Rows = rows;
            Columns = columns;
            SeqLength = seqLength;
            WeightPtr = weightPtr;
            GradPtr = gradPtr;
            Cache1Ptr = cache1Ptr;
            Cache2Ptr = cache2Ptr;
            CacheMPtr = cacheMPtr;
        }

        public int Rows;
        public int Columns;
        public int SeqLength;
        public IntPtr WeightPtr, GradPtr, Cache1Ptr, Cache2Ptr, CacheMPtr;
};
}