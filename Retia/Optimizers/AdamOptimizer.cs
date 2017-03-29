using System;
using Retia.Neural;

namespace Retia.Optimizers
{
    public class AdamOptimizer<T> : OptimizerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly float _b1, _b2;

        public AdamOptimizer(float learningRate = 1e-3f, float b1 = 0.9f, float b2 = 0.999f) : base(learningRate)
        {
            _b1 = b1;
            _b2 = b2;
        }

        public AdamOptimizer(AdamOptimizer<T> other) : base(other)
        {
            _b1 = other._b1;
            _b2 = other._b2;
        }

        public override void Optimize(NeuroWeight<T> weight)
        {
            weight.Timestep++;
            MathProvider.AdamUpdate(LearningRate, _b1, _b2, weight);
        }

        public override OptimizerBase<T> Clone()
        {
            return new AdamOptimizer<T>(this);
        }

        public override IntPtr CreateGpuOptimizer()
        {
            throw new NotSupportedException();
        }
    }
}