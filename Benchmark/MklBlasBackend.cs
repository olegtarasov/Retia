using System.Runtime.InteropServices;

namespace Retia.Mathematics
{
    public class MklBlasBackend : BlasBackendBase
    {
        public override unsafe void gemv(TransposeOptions trans, int m, int n, float alpha, float[] A, int lda, float[] x, int incx, float beta, float[] y, int incy)
        {
            fixed (float* pA = A, pX = x, pY = y)
            {
                cblas_sgemv((int)CblasOrder.CblasColMajor, (int)trans, m, n, alpha, pA, lda, pX, incx, beta, pY, incy);
            }
        }

        public override unsafe void ger(int m, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] A, int lda)
        {
            fixed (float* px = x, py = y, pA = A)
            {
                cblas_sger((int)CblasOrder.CblasColMajor, m, n, alpha, px, incx, py, incy, pA, lda);
            }
        }

        public override unsafe void axpy(int n, float alpha, float[] x, int incx, float[] y, int incy)
        {
            fixed (float* px = x, py = y)
            {
                cblas_saxpy(n, alpha, px, incx, py, incy);
            }
        }

        public override unsafe void gemm(TransposeOptions transA, TransposeOptions transB, int m, int n, int k, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc)
        {
            fixed (float* pA = A, pB = B, pc = C)
            {
                cblas_sgemm((int)CblasOrder.CblasColMajor, (int)transA, (int)transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pc, ldc);
            }
        }

        #region BLAS interface declaration

        private enum CblasOrder
        {
            CblasRowMajor = 101,
            CblasColMajor = 102
        }

        private enum CblasUplo
        {
            CblasUpper = 121,
            CblasLower = 122
        }

        private enum CblasDiag
        {
            CblasNonUnit = 131,
            CblasUnit = 132
        }

        private enum CblasSide
        {
            CblasLeft = 141,
            CblasRight = 142
        }

        private const string BLAS_DLL_S = "mkl_rt.dll";

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_sgemv(int order, int trans, int m, int n, float alpha, float* A,
            int lda, float* x,
            int incx, float beta, float* y, int incy);

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_sger(int order, int m, int n, float alpha, float* x,
            int incx, float* y, int incy, float* A, int lda);

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_saxpy(int n, float alpha, float* x,
            int incx, float* y, int incy);

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_sgemm(int order, int transA, int transB, int m, int n, int k,
            float alpha, float* A, int lda, float* B, int ldb,
            float beta, float* c, int ldc);
        #endregion
    }
}