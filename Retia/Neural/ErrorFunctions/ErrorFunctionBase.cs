using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Neural.ErrorFunctions
{
    public abstract class ErrorFunctionBase<T> : ICloneable<ErrorFunctionBase<T>> where T : struct, IEquatable<T>, IFormattable
    {
        protected readonly MathProviderBase<T> MathProvider = MathProvider<T>.Instance;

        public abstract double LayerError(Matrix<T> output, Matrix<T> target);
        public abstract List<Matrix<T>> BackpropagateError(List<Matrix<T>> outputs, List<Matrix<T>> targets);

        public abstract ErrorFunctionBase<T> Clone();
    }
}