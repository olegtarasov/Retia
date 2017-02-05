using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;

namespace Retia.Neural.Initializers
{
    public class ConstantMatrixInitializer : IMatrixInitializer
    {
        public float Value { get; set; } = 0.0f;

        public Matrix CreateMatrix(int rows, int columns)
        {
            return DenseMatrix.Create(rows, columns, Value);
        }

        public float[] CreateArray(int size)
        {
            return Enumerable.Repeat(Value, size).ToArray();
        }
    }
}