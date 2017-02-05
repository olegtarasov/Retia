using System;
using System.Runtime.InteropServices;
using System.Security;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural;

namespace Retia.Optimizers
{
	public class RMSPropOptimizer : OptimizerBase
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

		private RMSPropOptimizer(RMSPropOptimizer other) : base(other)
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

	    private unsafe void GravesRMSPropOptimizeSlow(NeuroWeight weight)
        {
            var gradient = weight.Gradient.AsColumnMajorArray();
            var cache1 = weight.Cache1.AsColumnMajorArray();
            var cache2 = weight.Cache2.AsColumnMajorArray();
            var cacheM = weight.CacheM.AsColumnMajorArray();
            var w = weight.Weight.AsColumnMajorArray();

            fixed (float* pGrad = gradient, pCache1 = cache1, pCache2 = cache2, pCacheM = cacheM, pWeight = w)
	        {
                ParallelFor.Instance.Execute(GravesRMSPropOptimize, gradient.Length, new void*[]{pGrad, pCache1, pCache2, pCacheM, pWeight});
	        }
        }

	    private unsafe void GravesRMSPropOptimize(int startIdx, int endIdx, void*[] ptrs)
	    {
            const float e = 1e-4f;

            float* pGrad = (float*)ptrs[0], pCache1 = (float*)ptrs[1], pCache2 = (float*)ptrs[2], pCacheM = (float*)ptrs[3], pWeight = (float*)ptrs[4];
            
            for (int i = startIdx; i < endIdx; i++)
	        {
                float* gradPtr = pGrad + i, cache1Ptr = pCache1 + i, cache2Ptr = pCache2 + i, cacheMPtr = pCacheM + i, weightPtr = pWeight + i;

                float grad = *gradPtr;
                float grad2 = grad * grad;

                float cache2 = *cache2Ptr = _decayRate * *cache2Ptr + (1 - _decayRate) * grad2;
                float cache1 = *cache1Ptr = _decayRate * *cache1Ptr + (1 - _decayRate) * grad;

                float k = (float)Math.Sqrt(cache2 - cache1 * cache1 + e);

                *cacheMPtr = _momentum * *cacheMPtr - LearningRate * grad / k;

                *weightPtr = *weightPtr + *cacheMPtr - LearningRate * _weightDecay * *weightPtr;
            }
	    }

	    /// <summary>
        /// RMSProp implementation proposed by A. Graves, with momentum.
        /// See http://arxiv.org/pdf/1308.0850v5.pdf, page 23
        /// </summary>
        /// <param name="weight">Weight matrix to optimize (incapsultes current gradient and all stored values needed for RMSProp)</param>
        //private unsafe void GravesRMSPropOptimize(NeuroWeight weight)
        //{
        //    var weightMatrix = weight.Weight;
        //    var gradient = weight.Gradient;
        //    var cache1 = weight.Cache1;
        //    var cache2 = weight.Cache2;
        //    var cacheM = weight.CacheM;

        //    var size = weightMatrix.Rows * weightMatrix.Cols;
        //    fixed (double* pGrad = (double[])gradient, pCache1 = (double[])cache1, pCache2 = (double[])cache2, pCacheM = (double[])cacheM, pWeight = (double[])weightMatrix)
        //    {
        //        FastGravesRMSPropUpdate(WeightDecay, LearningRate, DecayRate, Momentum, pWeight, pCache1, pCache2, pCacheM, pGrad, size);
        //    }
        //}

	    public override void Optimize(NeuroWeight weight)
        {
            //RMSPropOptimize(weight);
           GravesRMSPropOptimizeSlow(weight);
        }

		public override OptimizerBase Clone()
		{
			return new RMSPropOptimizer(this);
		}

	    public override OptimizerSpecBase CreateSpec()
	    {
	        return new RMSPropSpec(LearningRate, _momentum, _decayRate, _weightDecay);
	    }
	}
}