namespace Retia.Contracts
{
    public class RMSPropSpec : OptimizerSpecBase
    {
        public RMSPropSpec(float learningRate, float momentum, float decayRate, float weigthDecay) : base(learningRate)
        {
            Momentum = momentum;
            DecayRate = decayRate;
            WeigthDecay = weigthDecay;
        }

        public float Momentum { get; set; }
        public float DecayRate { get; set; }
        public float WeigthDecay { get; set; }

        public override OptimizerType OptimizerType => OptimizerType.RMSProp;
    }
}