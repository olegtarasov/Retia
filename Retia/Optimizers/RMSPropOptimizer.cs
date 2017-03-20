using System;
using System.Runtime.InteropServices;
using System.Security;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural;

namespace Retia.Optimizers
{
	public class RMSPropOptimizer<T> : OptimizerBase<T> where T : struct, IEquatable<T>, IFormattable
	{
	    [DllImport(Const.CudaDllName)]
	    private static extern IntPtr CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay);

        private readonly float _decayRate;
	    private readonly float _momentum;
	    private readonly float _weightDecay;

        public RMSPropOptimizer(float learningRate = 2e-3f, float decayRate = 0.95f, float weightDecay = 0.0f, float momentum = 0.9f) : base(learningRate)
		{
			LearningRate = learningRate;
			_decayRate = decayRate;
            _momentum = momentum;
            _weightDecay = weightDecay;
		}

		private RMSPropOptimizer(RMSPropOptimizer<T> other) : base(other)
		{
			LearningRate = other.LearningRate;
			_decayRate = other._decayRate;
		    _momentum = other._momentum;
		}

        public override void Optimize(NeuroWeight<T> weight)
        {
            MathProvider.GravesRmsPropUpdate(_weightDecay, LearningRate, _decayRate, _momentum, weight);
        }

		public override OptimizerBase<T> Clone()
		{
			return new RMSPropOptimizer<T>(this);
		}

        public override IntPtr CreateGpuOptimizer()
        {
            GpuOptimizerPtr = CreateRMSPropOptimizer(LearningRate, _momentum, _decayRate, _weightDecay);

            return GpuOptimizerPtr;
        }
    }
}