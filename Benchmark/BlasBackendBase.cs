namespace Retia.Mathematics
{
    public abstract class BlasBackendBase
    {
        public abstract void gemv(TransposeOptions trans, int m, int n, float alpha, float[] A, int lda, float[] x, int incx, float beta, float[] y, int incy);
        public abstract void ger(int m, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] A, int lda);
        public abstract void axpy(int n, float alpha, float[] x, int incx, float[] y, int incy);
        public abstract void gemm(TransposeOptions transA, TransposeOptions transB, int m, int n, int k, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc);
    }
}