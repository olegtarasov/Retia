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

        /// <summary>
        /// Default RMSProp optimization without momentum
        /// See http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf, slide 29
        /// </summary>
        /// <param name="weight">Weight matrix to optimize (incapsultes current gradient and all stored values needed for RMSProp)</param>
	    //private unsafe void RMSPropOptimize(NeuroWeight weight)
	    //{
     //       var weightMatrix = weight.Weight;
     //       var gradient = weight.Gradient;
     //       var cache2 = weight.Cache2;

     //       var size = weightMatrix.Rows * weightMatrix.Cols;
     //       fixed (double* pGrad = (double[])gradient, pCache2 = (double[])cache2, pWeight = (double[])weightMatrix)
     //       {
     //           FastRMSPropUpdate(LearningRate, DecayRate, pWeight, pCache2, pGrad, size);
     //       }
     //   }

	    public override void Optimize(NeuroWeight<T> weight)
        {
            //RMSPropOptimize(weight);
            MathProvider.GravesRMSPropUpdate(_weightDecay, LearningRate, _decayRate, _momentum, weight);
        }

		public override OptimizerBase<T> Clone()
		{
			return new RMSPropOptimizer<T>(this);
		}

	    public override OptimizerSpecBase CreateSpec()
	    {
	        return new RMSPropSpec(LearningRate, _momentum, _decayRate, _weightDecay);
	    }
	}
}