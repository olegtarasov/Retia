using System;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural;

namespace Retia.Optimizers
{
    /// <summary>
    /// Plain old SGD optimizer.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class SGDOptimizer<T> : OptimizerBase<T> where T : struct, IFormattable, IEquatable<T>
    {
        public SGDOptimizer(float learningRate) : base(learningRate)
        {
        }

        private SGDOptimizer(OptimizerBase<T> other) : base(other)
        {
        }

        public override void Optimize(NeuroWeight<T> weight)
        {
            weight.Weight.SubtractInplace(weight.Gradient.Multiply(MathProvider.Scalar(LearningRate)));
        }

        public override OptimizerSpecBase CreateSpec()
        {
            throw new NotSupportedException();
        }

        public override OptimizerBase<T> Clone()
        {
            return new SGDOptimizer<T>(this);
        }
    }
}