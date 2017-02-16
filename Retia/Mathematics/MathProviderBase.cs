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
        #region Generic ugliness

        public abstract T Scalar(float scalar);

        public abstract T Scalar(double scalar);

        public abstract T NaN();

        public abstract T[] Array(params float[] input);

        protected abstract bool AlmostEqual(T a, T b);

        #endregion

        #region Matrix operations

        public abstract void ClampMatrix(Matrix<T> matrix, T min, T max);

        public abstract Matrix<T> RandomMatrix(int rows, int cols, float min, float max);

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

        #endregion

        #region Optimization

        public abstract void AdagradUpdate(T learningRate, NeuroWeight<T> weight);

        public abstract void GravesRmsPropUpdate(float weightDecay, float learningRate, float decayRate, float momentum, NeuroWeight<T> weight);

        #endregion

        #region GRU layer

        public abstract void CalculateH(Matrix<T> H, Matrix<T> hCandidate, Matrix<T> z, Matrix<T> lastH);

        public abstract void ApplySigmoid2(Matrix<T> matrix1, Matrix<T> matrix2);

        public abstract void ApplyTanh(Matrix<T> matrix);

        #endregion

        #region Error functions

        public abstract List<int> SoftMaxChoice(Matrix<T> p, double T = 1.0);
        
        public abstract Matrix<T> SoftMaxNorm(Matrix<T> y, double T = 1.0);

        public abstract double CrossEntropyError(Matrix<T> p, Matrix<T> target);

        public abstract double MeanSquareError(Matrix<T> y, Matrix<T> target);

        #endregion

        #region Error backpropagation

        public abstract Matrix<T> BackPropagateMeanSquareError(Matrix<T> output, Matrix<T> target);

        public abstract Matrix<T> BackPropagateCrossEntropyError(Matrix<T> output, Matrix<T> target);

        public List<Matrix<T>> BackPropagateError(List<Matrix<T>> outputs, List<Matrix<T>> targets, Func<Matrix<T>, Matrix<T>, Matrix<T>> func)
        {
            if (outputs.Count != targets.Count || targets.Count == 0)
                throw new Exception("Not enough targets provided or not enough output states stored!");

            var sensitivities = new List<Matrix<T>>(outputs.Count);
            
            for (int i = 0; i < outputs.Count; i++)
            {
                var y = outputs[i];
                var target = targets[i];
                sensitivities.Add(func(y, target));
            }

            return sensitivities;
        }

        #endregion
    }
}