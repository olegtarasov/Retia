//using System;
//using System.Runtime.InteropServices;
//using Retia.Contracts;
//using Retia.Mathematics;
//using Retia.Neural;

//namespace Retia.Optimizers
//{
//	public class AdagradOptimizer : OptimizerBase
//    {
//        public AdagradOptimizer(float learningRate = 0.1f) : base(learningRate)
//        {
//            LearningRate = learningRate;
//        }

//		private AdagradOptimizer(AdagradOptimizer other) : base(other)
//		{
//		}

//        public override unsafe void Optimize(NeuroWeight weight)
//        {
//            var weightMatrix = weight.Weight.AsColumnMajorArray();
//            var gradient = weight.Gradient.AsColumnMajorArray();
//            var cache = weight.Cache2.AsColumnMajorArray();

//            fixed (float* pGrad = gradient, pMem = cache, pWeight = weightMatrix)
//            {
//                ParallelFor.Instance.Execute(AdagradUpdate, weightMatrix.Length, new void*[]{pGrad, pMem, pWeight});
//            }
//        }

//        private unsafe void AdagradUpdate(int startIdx, int endIdx, void*[] ptrs)
//        {
//            float* pGrad = (float*)ptrs[0], pMem = (float*)ptrs[1], pWeight = (float*)ptrs[2];

//            for (int i = startIdx; i < endIdx; i++)
//            {
//                float* gradPtr = pGrad + i, memPtr = pMem + i, weightPtr = pWeight + i;

//                float grad = *gradPtr;
//                float grad2 = grad * grad;
//                *memPtr += grad2;
//                float k = LearningRate / ((float)Math.Sqrt(*memPtr) + 1e-8f);
//                *weightPtr += -k * grad;
//            }
//        }

//        public override OptimizerBase Clone()
//		{
//			return new AdagradOptimizer(this);
//		}

//        public override OptimizerSpecBase CreateSpec()
//        {
//            throw new NotSupportedException();
//        }
//    }

//}