using System.Collections.Generic;
using System.Threading;

namespace Retia.Contracts
{
    public class GruLayerSpec : LayerSpecBase
    {
        public GruLayerSpec(int inputSize, int batchSize, int seqLen, int layers, int hSize, GruLayerWeights weights) : base(inputSize, batchSize, seqLen)
        {
            Layers = layers;
            HSize = hSize;
            Weights = new List<GruLayerWeights> {weights};
        }

        public int Layers { get; set; }
        public int HSize { get; set; }
        public List<GruLayerWeights> Weights { get; }

        public override LayerType LayerType => LayerType.Gru;
    }
}