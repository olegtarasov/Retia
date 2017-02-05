namespace Retia.Contracts
{
    public class SoftmaxLayerSpec : LayerSpecBase
    {
        public SoftmaxLayerSpec(int inputSize, int batchSize, int seqLen) : base(inputSize, batchSize, seqLen)
        {
        }

        public override LayerType LayerType => LayerType.Softmax;
    }
}