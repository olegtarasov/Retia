using System.Collections.Generic;

namespace Retia.Contracts
{
    public class LayeredNetSpec
    {
        public LayeredNetSpec(OptimizerSpecBase optimizer, int inputSize, int outputSize, int batchSize, int seqLen)
        {
            Optimizer = optimizer;
            InputSize = inputSize;
            BatchSize = batchSize;
            SeqLen = seqLen;
            OutputSize = outputSize;
        }

        public List<LayerSpecBase> Layers { get; } = new List<LayerSpecBase>();
        public OptimizerSpecBase Optimizer { get; set; }
        public int InputSize { get; set; }
        public int BatchSize { get; set; }
        public int OutputSize { get; set; }
        public int SeqLen { get; set; }
    }
}