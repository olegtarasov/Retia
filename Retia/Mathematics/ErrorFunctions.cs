using System;
using MathNet.Numerics.LinearAlgebra.Single;

namespace Retia.Mathematics
{
    public static class ErrorFunctions
    {
        public static float CrossEntropy(Matrix p, Matrix target)
        {
            if (p.ColumnCount != target.ColumnCount || p.RowCount != target.RowCount)
                throw new Exception("Matrix dimensions must agree!");
            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            float err = 0.0f;

            var rawP = p.AsColumnMajorArray();
            var rawT = target.AsColumnMajorArray();
            var n = rawP.Length;
            for (int i = 0; i < n; i++)
            {
                if(float.IsNaN(rawT[i]))
                    continue;
                err += rawT[i] * (float)Math.Log(rawP[i]);
            }
            //todo: should be fixed, we must take NaN cols into account
            return -err / p.ColumnCount;
        }

        public static float MeanSquare(Matrix y, Matrix target)
        {
            if (y.ColumnCount != target.ColumnCount || y.RowCount != target.RowCount)
                throw new Exception("Matrix dimensions must agree!");
            //E(y0, ... ,yn) = 0.5/n(target0-y0)^2 + ... + 0.5/n(target_n - y_n)^2
            float err = 0.0f;
            int n = y.ColumnCount * y.RowCount;
            int notNaN = 0;
            var rawY = y.AsColumnMajorArray();
            var rawT = target.AsColumnMajorArray();
            for (int i = 0; i < n; i++)
            {
                if(float.IsNaN(rawT[i]))
                    continue;
                notNaN++;
                float delta = rawT[i] - rawY[i];
                err += delta * delta;
            }

            return notNaN == 0 ? 0.0f : 0.5f * err / notNaN;
        }
    }
}