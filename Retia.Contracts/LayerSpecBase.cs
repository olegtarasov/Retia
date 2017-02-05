namespace Retia.Contracts
{
    public abstract class LayerSpecBase
    {
        protected LayerSpecBase(int inputSize, int batchSize, int seqLen)
        {
            InputSize = inputSize;
            BatchSize = batchSize;
            SeqLen = seqLen;
        }

        public int InputSize { get; set; }
        public int BatchSize { get; set; }
        public int SeqLen { get; set; }

        public abstract LayerType LayerType { get; }
    }
}