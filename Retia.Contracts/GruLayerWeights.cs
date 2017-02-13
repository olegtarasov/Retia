using MathNet.Numerics.LinearAlgebra;

namespace Retia.Contracts
{
    public class GruLayerWeights
    {
        public Matrix<float> Wxr { get; set; }
        public Matrix<float> Wxz { get; set; }
        public Matrix<float> Wxh { get; set; }
                     
        public Matrix<float> Whr { get; set; }
        public Matrix<float> Whz { get; set; }
        public Matrix<float> Whh { get; set; }
                     
        public Matrix<float> bxr { get; set; }
        public Matrix<float> bxz { get; set; }
        public Matrix<float> bxh { get; set; }
                     
        public Matrix<float> bhr { get; set; }
        public Matrix<float> bhz { get; set; }
        public Matrix<float> bhh { get; set; }
    }
}