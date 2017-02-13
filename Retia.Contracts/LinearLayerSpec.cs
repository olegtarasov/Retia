using MathNet.Numerics.LinearAlgebra;

namespace Retia.Contracts
{
    public class LinearLayerSpec : LayerSpecBase
    {
        public LinearLayerSpec(int inputSize, int batchSize, int seqLen, int outSize, Matrix<float> w, Matrix<float> b) : base(inputSize, batchSize, seqLen)
        {
            OutSize = outSize;
            W = w;
            this.b = b;
        }

        public int OutSize { get; set; }
        public Matrix<float> W { get; set; }
        public Matrix<float> b { get; set; }

        public override LayerType LayerType => LayerType.Linear;
    }
}