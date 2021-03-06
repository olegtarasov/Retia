using System;
using Retia.Interop;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural;

namespace Retia.Optimizers
{
	public abstract class OptimizerBase<T> : ICloneable<OptimizerBase<T>>, IOptimizer where T : struct, IEquatable<T>, IFormattable
	{
	    protected readonly MathProviderBase<T> MathProvider = MathProvider<T>.Instance;

	    protected IntPtr GpuOptimizerPtr = IntPtr.Zero;

	    private float _learningRate;

	    protected OptimizerBase(float learningRate)
	    {
	        LearningRate = learningRate;
	    }

	    protected OptimizerBase(OptimizerBase<T> other)
	    {
	        LearningRate = other.LearningRate;
	    }

	    public float LearningRate
	    {
	        get { return _learningRate; }
	        set
	        {
	            _learningRate = value;
	            if (GpuOptimizerPtr != IntPtr.Zero)
	            {
                    GpuInterface.SetLearningRate(GpuOptimizerPtr, value);
	            }
	        }
	    }

	    public void DestroyGpuOptimizer()
	    {
	        if (GpuOptimizerPtr != IntPtr.Zero)
	        {
                GpuInterface.DestroyOptimizer(GpuOptimizerPtr);
                GpuOptimizerPtr = IntPtr.Zero;
	        }
	    }

	    public abstract OptimizerBase<T> Clone();
	    public abstract IntPtr CreateGpuOptimizer();

	    public abstract void Optimize(NeuroWeight<T> weight);
	}
}