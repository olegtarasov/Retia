using System;
using Retia.Contracts;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural;

namespace Retia.Optimizers
{
	public abstract class OptimizerBase<T> : ICloneable<OptimizerBase<T>> where T : struct, IEquatable<T>, IFormattable
	{
	    protected MathProviderBase<T> MathProvider = MathProvider<T>.Instance;

	    protected OptimizerBase(float learningRate)
	    {
	        LearningRate = learningRate;
	    }

	    protected OptimizerBase(OptimizerBase<T> other)
	    {
	        LearningRate = other.LearningRate;
	    }

	    public abstract void Optimize(NeuroWeight<T> weight);
	    public abstract OptimizerSpecBase CreateSpec();

		public abstract OptimizerBase<T> Clone();
	    public float LearningRate { get; set; }
	}
}