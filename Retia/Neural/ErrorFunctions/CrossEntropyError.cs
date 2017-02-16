using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Retia.Neural.ErrorFunctions
{
    public class CrossEntropyError<T> : ErrorFunctionBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public override double LayerError(Matrix<T> output, Matrix<T> target)
        {
            return MathProvider.CrossEntropyError(output, target);
        }

        public override List<Matrix<T>> BackpropagateError(List<Matrix<T>> outputs, List<Matrix<T>> targets)
        {
            return MathProvider.BackPropagateError(outputs, targets, MathProvider.BackPropagateCrossEntropyError);
        }

        public override ErrorFunctionBase<T> Clone()
        {
            return new CrossEntropyError<T>();
        }
    }
}