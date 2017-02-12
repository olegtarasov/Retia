using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public abstract class NeuroLayer<T> : ICloneable<NeuroLayer<T>>, IFileWritable where T : struct, IEquatable<T>, IFormattable
    {
        protected int BatchSize;
        protected int SeqLen;

        protected List<Matrix<T>> Inputs = new List<Matrix<T>>();
        protected List<Matrix<T>> Outputs = new List<Matrix<T>>();

        protected readonly MathProviderBase<T> MathProvider = MathProvider<T>.Instance;

        protected NeuroLayer()
        {
        }

        protected NeuroLayer(NeuroLayer<T> other)
        {
            BatchSize = other.BatchSize;
            SeqLen = other.SeqLen;
            Inputs = other.Inputs.Select(x => x.CloneMatrix()).ToList();
            Outputs = other.Outputs.Select(x => x.CloneMatrix()).ToList();
        }

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }
        public abstract int TotalParamCount { get; }
        public virtual Matrix<T>[] InternalState { get;  set; }

        public abstract NeuroLayer<T> Clone();

        public abstract void Save(Stream s);

        public abstract void Optimize(OptimizerBase<T> optimizer);

        /// <summary>
        ///     Forward layer step
        /// </summary>
        /// <param name="input">Input matrix</param>
        /// <param name="inTraining">Store states for back propagation</param>
        /// <returns>Layer output</returns>
        public abstract Matrix<T> Step(Matrix<T> input, bool inTraining = false);

        public abstract void ResetMemory();
        public abstract void ResetOptimizer();
        public abstract void InitSequence();
        public abstract void ClampGrads(float limit);
        public abstract LayerSpecBase CreateSpec();

        /// <summary>
        /// Converts layer state to a vector of doubles.
        /// </summary>
        /// <returns>Layer vector state.</returns>
        public abstract void ToVectorState(T[] destination, ref int idx, bool grad=false);

        /// <summary>
        /// Modifies layer state from a vector of doubles.
        /// </summary>
        public abstract void FromVectorState(T[] vector, ref int idx);


        /// <summary>
        ///     Propagates next layer sensitivity to input, accumulating gradients for optimization
        /// </summary>
        /// <param name="outSens">Sequence of sensitivity matrices of next layer</param>
        /// <param name="needInputSens">Calculate input sensitivity for further propagation</param>
        /// <returns></returns>
        public virtual List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true)
        {
            return outSens;
        }


        /// <summary>
        ///     Propagates next layer sensitivity to input, calculating input sensitivity matrix
        /// </summary>
        /// <param name="outSens">Sequence of sensitivity matrices of next layer</param>
        /// <returns></returns>
        public virtual List<Matrix<T>> PropagateSensitivity(List<Matrix<T>> outSens)
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

        /// <summary>
        ///     Calculates matched layer error.
        /// </summary>
        /// <param name="y">Layer output</param>
        /// <param name="target">Layer target</param>
        /// <returns></returns>
        public virtual double LayerError(Matrix<T> y, Matrix<T> target)
        {
            return 0.0;
        }

        public virtual void SetParam(int i, T value)
        {
            int refInd = 0;
            var state = new T[TotalParamCount];
            ToVectorState(state, ref refInd);
            state[i] = value;
            refInd = 0;
            FromVectorState(state, ref refInd);
        }

        /// <summary>
        ///     Calculates matched error (out-target) and propagates it through layer to inputs
        /// </summary>
        /// <param name="targets">Sequence of targets</param>
        public virtual List<Matrix<T>> ErrorPropagate(List<Matrix<T>> targets)
        {
            return MathProvider.ErrorPropagate(Outputs, targets, SeqLen, BatchSize);
        }

        public void Save(string filename)
        {
            this.SaveObject(filename);
        }

        public T GetParam(int i, bool grad = false)
        {
            int refInd=0;
            var state = new T[TotalParamCount];
            ToVectorState(state, ref refInd, grad);
            return state[i];
        }

        protected virtual void Initialize()
        {
        }

        /// <summary>
        ///     Get value of layer output derrivative with respect to input (dO/dI of [batch])
        /// </summary>
        /// <param name="input">Input value matrix</param>
        /// <param name="output">Output value matrix</param>
        /// <param name="batch">Batch index</param>
        /// <param name="i">Input index</param>
        /// <param name="o">Output index</param>
        /// <returns>Derivative value</returns>
        protected virtual T Derivative(Matrix<T> input, Matrix<T> output, int batch, int i, int o)
        {
            return default(T);
        }

        internal void Initialize(int batchSize, int seqLen)
        {
            BatchSize = batchSize;
            SeqLen = seqLen;

            Initialize();
        }

        private void CalcSens(int step, int batch, Matrix<T> iSens, Matrix<T> outSens)
        {
            // TODO: Support this
            throw new NotSupportedException();
            //for (int i = 0; i < InputSize; i++)
            //{
            //    for (int o = 0; o < OutputSize; o++)
            //    {
            //        iSens[i, batch] += Derivative(Inputs[step], Outputs[step], batch, i, o) * outSens[o, batch];
            //    }
            //}
        }

        //}
        //    return d[o, batch];

        //    var d = (0.5f / delta) * (p - n);
        //    var n = Step(nInput);

        //    var p = Step(pInput);
        //    nInput[i, batch] -= delta;
        //    pInput[i, batch] += delta;
        //    var nInput = input.CloneMatrix();

        //    var pInput = input.CloneMatrix();
        //    const float delta = 1e-5f;
        //{

        //public virtual float NumDerrivative(Matrix input, Matrix output, int batch, int i, int o)
    }
}