using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Neural.ErrorFunctions
{
    /// <summary>
    /// Base class for the error function. Error function consists of a forward pass function
    /// and a backpropagation function.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class ErrorFunctionBase<T> : ICloneable<ErrorFunctionBase<T>> where T : struct, IEquatable<T>, IFormattable
    {
        protected readonly MathProviderBase<T> MathProvider = MathProvider<T>.Instance;

        /// <summary>
        /// Gets an error value for single output with respect to specified target.
        /// </summary>
        /// <param name="output">Output matrix.</param>
        /// <param name="target">Target matrix.</param>
        /// <returns>Error value.</returns>
        public abstract double GetError(Matrix<T> output, Matrix<T> target);

        /// <summary>
        /// Propagates the error backwards.
        /// </summary>
        /// <param name="outputs">Ouput matrix sequence.</param>
        /// <param name="targets">Target matrix sequence.</param>
        /// <returns>The sequence of error sensitivities.</returns>
        public abstract List<Matrix<T>> BackpropagateError(List<Matrix<T>> outputs, List<Matrix<T>> targets);

        public abstract ErrorFunctionBase<T> Clone();
    }
}