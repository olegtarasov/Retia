//using System;
//using System.Runtime.InteropServices;
//using System.Threading.Tasks;
//using MathNet.Numerics.LinearAlgebra;
//using MathNet.Numerics.LinearAlgebra.Single;
//using Retia.Helpers;

//namespace Retia.Mathematics
//{
//    public static class ActivationFuncs<T> where T : struct, IEquatable<T>, IFormattable
//    {
//        private static bool _isFloat = typeof(T) == typeof(float);

//        [DllImport("FastFuncs")]
//        private static extern void ApplySigmoid2D(IntPtr a, IntPtr b, int n);

//        [DllImport("FastFuncs")]
//        private static extern void ApplySigmoid2S(IntPtr a, IntPtr b, int n);

//        [DllImport("FastFuncs")]
//        private static extern void ApplyTanhD(IntPtr matrix, int n);

//        [DllImport("FastFuncs")]
//        private static extern void ApplyTanhS(IntPtr matrix, int n);

//        public static void ApplySigmoid2(Matrix<T> matrix1, Matrix<T> matrix2)
//        {
//            using (var ptrs = new MatrixPointers<T>(matrix1, matrix2))
//            {
//                if (_isFloat)
//                    ApplySigmoid2S(ptrs[0], ptrs[1], matrix1.Length());
//                else
//                    ApplySigmoid2D(ptrs[0], ptrs[1], matrix1.Length());
//            }
//        }

//        public static void ApplyTanh(Matrix<T> matrix)
//        {
//            using (var ptrs = new MatrixPointers<T>(matrix))
//            {
//                if (_isFloat)
//                    ApplyTanhS(ptrs[0], matrix.Length());
//                else
//                    ApplyTanhD(ptrs[0], matrix.Length());
//            }
//        }
//    }
//}