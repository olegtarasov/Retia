using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;

namespace Retia.Neural.Initializers
{
    public interface IMatrixInitializer
    {
        Matrix CreateMatrix(int rows, int columns);
    }
}