using MathNet.Numerics.LinearAlgebra.Single;

namespace Retia.Contracts
{
    public class LinearLayerSpec : LayerSpecBase
    {
        public LinearLayerSpec(int inputSize, int batchSize, int seqLen, int outSize, Matrix w, Matrix b) : base(inputSize, batchSize, seqLen)
        {
            OutSize = outSize;
            W = w;
            this.b = b;
        }

        public int OutSize { get; set; }
        public Matrix W { get; set; }
        public Matrix b { get; set; }

        public override LayerType LayerType => LayerType.Linear;
    }
}