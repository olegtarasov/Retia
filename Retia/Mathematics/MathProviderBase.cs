using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Retia.Neural;

namespace Retia.Mathematics
{
    /// <summary>
    /// Base class for the math provider.
    /// </summary>
    /// <typeparam name="T">Data type</typeparam>
    public abstract class MathProviderBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        static MathProviderBase()
        {
            // Try to use MKL as soon as possible
            Control.TryUseNativeMKL();
        }

        #region Generic ugliness

        /// <summary>
        /// Converts float value to T.
        /// </summary>
        public abstract T Scalar(float scalar);

        /// <summary>
        /// Converts double value to T.
        /// </summary>
        public abstract T Scalar(double scalar);

        /// <summary>
        /// Gets NaN value for T.
        /// </summary>
        public abstract T NaN();

        /// <summary>
        /// Converts an array of float values to an array of T.
        /// </summary>
        public abstract T[] Array(params float[] input);

        /// <summary>
        /// Tests two values of T for equality with error margin of 10e-7.
        /// </summary>
        protected abstract bool AlmostEqual(T a, T b);

        #endregion

        #region Matrix operations

        /// <summary>
        /// Clips matrix values to the range of [min;max].
        /// </summary>
        /// <param name="matrix">Matrix to clamp.</param>
        /// <param name="min">Minimum value.</param>
        /// <param name="max">Maximum value.</param>
        public abstract void ClampMatrix(Matrix<T> matrix, T min, T max);

        /// <summary>
        /// Creates a random matrix in the range of [min;max).
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        /// <param name="min">Minimum value.</param>
        /// <param name="max">Maximum value.</param>
        public abstract Matrix<T> RandomMatrix(int rows, int cols, float min, float max);

        /// <summary>
        /// Creates a random mask matrix. For each element roll the dice in range of [0..1) and
        /// if the value is less than <see cref="trueProb"/> set the element to 1. Otherwise set
        /// to 0.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        /// <param name="trueProb">Probability of setting an element to 1, range [0..1].</param>
        public abstract Matrix<T> RandomMaskMatrix(int rows, int cols, float trueProb);

        /// <summary>
        /// Saves a matrix to a stream.
        /// </summary>
        /// <param name="matrix">Matrix to save.</param>
        /// <param name="stream">Stream.</param>
        public abstract void SaveMatrix(Matrix<T> matrix, Stream stream);

        /// <summary>
        /// Tests two matrices for equality with error margin of 10e-7.
        /// </summary>
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

        /// <summary>
        /// Performs Adam update on a weight.
        /// </summary>
        /// <remarks>See https://arxiv.org/pdf/1412.6980.pdf </remarks>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="b1">Decay rate of first order MAV.</param>
        /// <param name="b2">Decay rate of second order MAV.</param>
        /// <param name="weight">Weight.</param>
        public abstract void AdamUpdate(float learningRate, float b1, float b2, NeuroWeight<T> weight);

        /// <summary>
        /// Performs Adagrad update on a weight.
        /// </summary>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="weight">Weight.</param>
        public abstract void AdagradUpdate(T learningRate, NeuroWeight<T> weight);

        /// <summary>
        /// Performs Graves' version of RMSProp update on a weight.
        /// </summary>
        /// <remarks>See http://arxiv.org/pdf/1308.0850v5.pdf, page 23</remarks>
        /// <param name="weightDecay">Weight decay rate.</param>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="decayRate">Decay rate.</param>
        /// <param name="momentum">Momentum.</param>
        /// <param name="weight">Weight.</param>
        public abstract void GravesRmsPropUpdate(float weightDecay, float learningRate, float decayRate, float momentum, NeuroWeight<T> weight);

        #endregion

        #region GRU layer

        /// <summary>
        /// Calculates the final hidden state for a GRU layer.
        /// </summary>
        public abstract void CalculateH(Matrix<T> H, Matrix<T> hCandidate, Matrix<T> z, Matrix<T> lastH);

        /// <summary>
        /// Applies sigmoid function to one matrix.
        /// </summary>
        public abstract void ApplySigmoid(Matrix<T> matrix);

        /// <summary>
        /// Applies sigmoid function to two matrices in parallel.
        /// </summary>
        public abstract void ApplySigmoid2(Matrix<T> matrix1, Matrix<T> matrix2);

        /// <summary>
        /// Applies tanh function to a matrix.
        /// </summary>
        public abstract void ApplyTanh(Matrix<T> matrix);

        #endregion

        #region Error functions

        /// <summary>
        /// Chooses an integer in range [0..RowCount] for each column in a matrix according to SoftMax probabilities.
        /// </summary>
        /// <param name="p">The matrix of SoftMax probabilities.</param>
        /// <param name="T">Scaling factor.</param>
        public abstract List<int> SoftMaxChoice(Matrix<T> p, double T = 1.0);
        
        /// <summary>
        /// Normalizes matrix values with SoftMax and returns the result.
        /// </summary>
        /// <param name="y">Matrix to nornalize.</param>
        /// <param name="T">Scaling factor.</param>
        public abstract Matrix<T> SoftMaxNorm(Matrix<T> y, double T = 1.0);

        /// <summary>
        /// Calculates the cross-entropy error between output <see cref="p"/> and target <see cref="target"/>.
        /// </summary>
        /// <remarks>
        /// E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
        /// </remarks>
        /// <param name="p">Output probabilities.</param>
        /// <param name="target">Target probabilities.</param>
        /// <returns>Cross-entropy error.</returns>
        public abstract double CrossEntropyError(Matrix<T> p, Matrix<T> target);

        /// <summary>
        /// Calculates the mean squared error between output <see cref="y"/> and target <see cref="target"/>.
        /// </summary>
        /// <remarks>
        /// E(y0, ... ,yn) = 0.5/n(target0-y0)^2 + ... + 0.5/n(target_n - y_n)^2
        /// </remarks>
        /// <param name="y">Ouput values.</param>
        /// <param name="target">Target values.</param>
        /// <returns>Mean squared error.</returns>
        public abstract double MeanSquareError(Matrix<T> y, Matrix<T> target);

        #endregion

        #region Error backpropagation

        /// <summary>
        /// Propagates mean square error backwards.
        /// </summary>
        /// <param name="output">Output values.</param>
        /// <param name="target">Target values.</param>
        /// <returns>Error sensitivities.</returns>
        public abstract Matrix<T> BackPropagateMeanSquareError(Matrix<T> output, Matrix<T> target);

        /// <summary>
        /// Propagates cross entropy error backwards.
        /// </summary>
        /// <param name="output">Output values.</param>
        /// <param name="target">Target values.</param>
        /// <returns>Error sensitivities.</returns>
        public abstract Matrix<T> BackPropagateCrossEntropyError(Matrix<T> output, Matrix<T> target);

        /// <summary>
        /// Backpropagates the sequence of errors with selected function.
        /// </summary>
        /// <param name="outputs">Output sequence.</param>
        /// <param name="targets">Target sequence.</param>
        /// <param name="func">Bacpropagation function.</param>
        /// <returns>The sequence of error sensitivities.</returns>
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