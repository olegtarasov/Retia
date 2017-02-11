using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Retia.Mathematics
{
    public static class ActivationFuncs
    {
        public static unsafe void ApplySigmoid2(Matrix matrix1, Matrix matrix2)
        {
            var a1 = matrix1.AsColumnMajorArray();
            var a2 = matrix2.AsColumnMajorArray();

            fixed (double* pArray1 = a1, pArray2 = a2)
            {
                ParallelFor.Instance.Execute(ApplySigmoid2, a1.Length, new void*[] {pArray1, pArray2});
            }
        }

        private static unsafe void ApplySigmoid2(int startIdx, int endIdx, void*[] ptrs)
        {
            float* startPtr1 = (float*)ptrs[0], startPtr2 = (float*)ptrs[1];

            for (int i = startIdx; i < endIdx; i++)
            {
                ApplySigmoid(startPtr1 + i);
                ApplySigmoid(startPtr2 + i);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ApplySigmoid(float* ptr)
        {
            *ptr = 1.0f / (1 + (float)Math.Exp(-(*ptr)));
        }

        public static unsafe void ApplyTanh(Matrix matrix)
        {
            var a = matrix.AsColumnMajorArray();
            fixed (double* pArray = a)
            {
                ParallelFor.Instance.Execute(ApplyTanh, a.Length, new void*[] {pArray});
            }
        }

        private static unsafe void ApplyTanh(int startIdx, int endIdx, void*[] ptrs)
        {
            float* startPtr = (float*)ptrs[0];

            for (int i = startIdx; i < endIdx; i++)
            {
                float* ptr = startPtr + i;
                *ptr = (float)Math.Tanh(*ptr);
            }
        }
    }
}