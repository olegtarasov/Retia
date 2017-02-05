namespace Retia.Contracts
{
    public abstract class OptimizerSpecBase
    {
        protected OptimizerSpecBase(float learningRate)
        {
            LearningRate = learningRate;
        }

        public float LearningRate { get; set; }

        public abstract OptimizerType OptimizerType { get; }
    }
}