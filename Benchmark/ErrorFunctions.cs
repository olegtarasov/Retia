using System;
using Retia.Mathematics;

//using df = System.Double;
using df = System.Single;

namespace Benchmark
{
    public static class ErrorFunctions
    {
        public static df CrossEntropy(Matrix p, Matrix target)
        {
            if (p.Cols != target.Cols || p.Rows != target.Rows)
                throw new Exception("Matrix dimensions must agree!");
            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            df err = 0.0f;

            var rawP = (df[])p;
            var rawT = (df[])target;
            var n = rawP.Length;
            for (int i = 0; i < n; i++)
            {
                if(df.IsNaN(rawT[i]))
                    continue;
                err += rawT[i] * (float)Math.Log(rawP[i]);
            }
            //todo: should be fixed, we must take NaN cols into account
            return -err / p.Cols;
        }

        public static df MeanSquare(Matrix y, Matrix target)
        {
            if (y.Cols != target.Cols || y.Rows != target.Rows)
                throw new Exception("Matrix dimensions must agree!");
            //E(y0, ... ,yn) = 0.5/n(target0-y0)^2 + ... + 0.5/n(target_n - y_n)^2
            df err = 0.0f;
            int notNaN = 0;
            var rawY = (df[])y;
            var rawT = (df[])target;
            for (int i = 0; i < y.Length; i++)
            {
                if(df.IsNaN(rawT[i]))
                    continue;
                notNaN++;
                df delta = rawT[i] - rawY[i];
                err += delta * delta;
            }

            return notNaN == 0 ? 0.0f : 0.5f * err / notNaN;
        }
    }
}