using System;
using Retia.Neural;

namespace Retia.Optimizers
{
    public class AdagradOptimizer<T> : OptimizerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public AdagradOptimizer(float learningRate = 0.1f) : base(learningRate)
        {
            LearningRate = learningRate;
        }

        private AdagradOptimizer(AdagradOptimizer<T> other) : base(other)
        {
        }

        public override void Optimize(NeuroWeight<T> weight)
        {
            MathProvider.AdagradUpdate(MathProvider.Scalar(LearningRate), weight);
        }

        public override OptimizerBase<T> Clone()
        {
            return new AdagradOptimizer<T>(this);
        }

        public override IntPtr CreateGpuOptimizer()
        {
            throw new NotSupportedException();
        }
    }

}