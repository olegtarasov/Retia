using MathNet.Numerics.LinearAlgebra.Single;

namespace Retia.Contracts
{
    public class GruLayerWeights
    {
        public Matrix Wxr { get; set; }
        public Matrix Wxz { get; set; }
        public Matrix Wxh { get; set; }
               
        public Matrix Whr { get; set; }
        public Matrix Whz { get; set; }
        public Matrix Whh { get; set; }
               
        public Matrix bxr { get; set; }
        public Matrix bxz { get; set; }
        public Matrix bxh { get; set; }
               
        //public Matrix bhr { get; set; }
        //public Matrix bhz { get; set; }
        //public Matrix bhh { get; set; }
    }
}