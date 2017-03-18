using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace Retia.Neural.Layers
{
    /// <summary>
    /// Base layer that supports backpropagation of sensitivities with custom element-wise
    /// derivative fuction.
    /// </summary>
    public abstract class DerivativeLayerBase<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly bool _isSingle = typeof(T) == typeof(float);

        protected DerivativeLayerBase()
        {
        }

        protected DerivativeLayerBase(LayerBase<T> other) : base(other)
        {
        }

        protected DerivativeLayerBase(BinaryReader reader) : base(reader)
        {
        }

        /// <summary>
        ///     Get value of layer output derrivative with respect to input (dO/dI of [batch]). Single precision version.
        /// </summary>
        /// <param name="input">Input value matrix</param>
        /// <param name="output">Output value matrix</param>
        /// <param name="batch">Batch index</param>
        /// <param name="i">Input index</param>
        /// <param name="o">Output index</param>
        /// <returns>Derivative value</returns>
        protected abstract float DerivativeS(Matrix<float> input, Matrix<float> output, int batch, int i, int o);

        /// <summary>
        ///     Get value of layer output derrivative with respect to input (dO/dI of [batch]). Double precision version.
        /// </summary>
        /// <param name="input">Input value matrix</param>
        /// <param name="output">Output value matrix</param>
        /// <param name="batch">Batch index</param>
        /// <param name="i">Input index</param>
        /// <param name="o">Output index</param>
        /// <returns>Derivative value</returns>
        protected abstract double DerivativeD(Matrix<double> input, Matrix<double> output, int batch, int i, int o);

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true, bool clearGrad = true)
        {
            if (Outputs.Count != Inputs.Count)
                throw new Exception("Backprop was not initialized (empty state sequence)");
            if (Inputs.Count == 0)
                throw new Exception("Empty inputs history, nothing to propagate!");
            if (outSens.Count != Inputs.Count)
                throw new Exception("Not enough sensitivies in list!");

            return PropagateSensitivity(outSens);
        }

        /// <summary>
        ///     Propagates next layer sensitivity to input, calculating input sensitivity matrix
        /// </summary>
        /// <param name="outSens">Sequence of sensitivity matrices of next layer</param>
        /// <returns></returns>
        protected List<Matrix<T>> PropagateSensitivity(List<Matrix<T>> outSens)
        {
            var iSensList = new List<Matrix<T>>(Inputs.Count);
            for (int step = 0; step < Inputs.Count; step++)
            {
                var oSens = outSens[step];
                var iSens = Matrix<T>.Build.Dense(InputSize, BatchSize);
                for (int b = 0; b < BatchSize; b++)
                    CalcSens(step, b, iSens, oSens);
                iSensList.Add(iSens);
            }
            return iSensList;
        }

        private void CalcSens(int step, int batch, Matrix<T> iSens, Matrix<T> outSens)
        {
            if (_isSingle)
            {
                var isens = iSens as Matrix<float>;
                var osens = outSens as Matrix<float>;
                for (int i = 0; i < InputSize; i++)
                {
                    for (int o = 0; o < OutputSize; o++)
                    {
                        isens[i, batch] += DerivativeS(Inputs[step] as Matrix<float>, Outputs[step] as Matrix<float>, batch, i, o) * osens[o, batch];
                    }
                }
            }
            else
            {
                var isens = iSens as Matrix<double>;
                var osens = outSens as Matrix<double>;
                for (int i = 0; i < InputSize; i++)
                {
                    for (int o = 0; o < OutputSize; o++)
                    {
                        isens[i, batch] += DerivativeD(Inputs[step] as Matrix<double>, Outputs[step] as Matrix<double>, batch, i, o) * osens[o, batch];
                    }
                }
            }
        }
    }
}