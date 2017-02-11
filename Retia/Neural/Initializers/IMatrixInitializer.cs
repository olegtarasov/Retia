using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Mathematics;

namespace Retia.Neural.Initializers
{
    public interface IMatrixInitializer
    {
        Matrix CreateMatrix(int rows, int columns);
    }
}