using Retia.Contracts;
using Retia.Integration;
using Retia.Neural;

namespace Retia.Optimizers
{
	public abstract class OptimizerBase : ICloneable<OptimizerBase>
	{
	    protected OptimizerBase(float learningRate)
	    {
	        LearningRate = learningRate;
	    }

	    protected OptimizerBase(OptimizerBase other)
	    {
	        LearningRate = other.LearningRate;
	    }

	    public abstract void Optimize(NeuroWeight weight);
	    public abstract OptimizerSpecBase CreateSpec();

		public abstract OptimizerBase Clone();
	    public float LearningRate { get; set; }
	}
}