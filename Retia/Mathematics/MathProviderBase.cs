using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public abstract class MathProviderBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public abstract List<int> SoftMaxChoice(Matrix<T> p, double T = 1.0);

        public abstract T Scalar(float scalar);

        public abstract T Scalar(double scalar);

        public abstract T NaN();

        public abstract void GravesRmsPropUpdate(float weightDecay, float learningRate, float decayRate, float momentum, NeuroWeight<T> weight);

        public abstract void CalculateH(Matrix<T> H, Matrix<T> hCandidate, Matrix<T> z, Matrix<T> lastH);
        
        public abstract Matrix<T> SoftMaxNorm(Matrix<T> y, double T = 1.0);

        public abstract void ApplySigmoid2(Matrix<T> matrix1, Matrix<T> matrix2);

        public abstract void ApplyTanh(Matrix<T> matrix);

        public abstract double CrossEntropy(Matrix<T> p, Matrix<T> target);

        public abstract double MeanSquare(Matrix<T> y, Matrix<T> target);

        protected abstract Matrix<T> PropagateSingleError(Matrix<T> y, Matrix<T> target, int batchSize);

        protected abstract bool AlmostEqual(T a, T b);

        public List<Matrix<T>> ErrorPropagate(List<Matrix<T>> outputs, List<Matrix<T>> targets, int seqLen, int batchSize)
        {
            if (outputs.Count != targets.Count || targets.Count == 0)
                throw new Exception("Not enough targets provided or not enough output states stored!");

            var sensitivities = new List<Matrix<T>>(seqLen);
            
            for (int i = 0; i < seqLen; i++)
            {
                var y = outputs[i];
                var target = targets[i];
                sensitivities.Add(PropagateSingleError(y, target, batchSize));
            }

            return sensitivities;
        }

        public bool MatricesEqual(Matrix<T> matrix, Matrix<T> other)
        {
            if (ReferenceEquals(matrix, other))
            {
                return true;
            }

            var m1 = matrix.AsColumnMajorArray();
            var m2 = other.AsColumnMajorArray();

            if (ReferenceEquals(m1, m2))
            {
                return true;
            }

            if (m1.Length != m2.Length)
            {
                return false;
            }

            for (int i = 0; i < m1.Length; i++)
            {
                if (!AlmostEqual(m1[i], m2[i]))
                {
                    return false;
                }
            }

            return true;
        }
    }
}