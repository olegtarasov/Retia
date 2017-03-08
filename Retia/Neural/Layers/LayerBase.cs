using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural.ErrorFunctions;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public abstract class LayerBase<T> : ICloneable<LayerBase<T>>, IFileWritable where T : struct, IEquatable<T>, IFormattable
    {
        protected readonly MathProviderBase<T> MathProvider = MathProvider<T>.Instance;
        protected int BatchSize;
        protected int SeqLen;

        protected List<Matrix<T>> Inputs = new List<Matrix<T>>();
        protected List<Matrix<T>> Outputs = new List<Matrix<T>>();

        protected LayerBase()
        {
        }

        protected LayerBase(LayerBase<T> other)
        {
            BatchSize = other.BatchSize;
            SeqLen = other.SeqLen;
            Inputs = other.Inputs.Select(x => x.CloneMatrix()).ToList();
            Outputs = other.Outputs.Select(x => x.CloneMatrix()).ToList();
            ErrorFunction = other.ErrorFunction?.Clone();
        }

        protected LayerBase(BinaryReader reader)
        {
            BatchSize = reader.ReadInt32();
            SeqLen = reader.ReadInt32();

            bool hasError = reader.ReadBoolean();
            if (hasError)
            {
                string errorType = reader.ReadString();
                ErrorFunction = (ErrorFunctionBase<T>)Activator.CreateInstance(Type.GetType(errorType));
            }
        }

        public ErrorFunctionBase<T> ErrorFunction { get; set; }

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }
        public abstract int TotalParamCount { get; }
        public virtual Matrix<T>[] InternalState { get;  set; }

        public abstract LayerBase<T> Clone();

        public virtual void Save(Stream s)
        {
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(BatchSize);
                writer.Write(SeqLen);
                writer.Write(ErrorFunction != null);

                if (ErrorFunction != null)
                {
                    writer.Write(ErrorFunction.GetType().FullName);
                }
            }
        }

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
        ///     Calculates matched layer error.
        /// </summary>
        /// <param name="y">Layer output</param>
        /// <param name="target">Layer target</param>
        /// <returns></returns>
        public virtual double LayerError(Matrix<T> y, Matrix<T> target)
        {
            if (ErrorFunction == null)
            {
                throw new InvalidOperationException("Layer error function is not specified!");
            }

            return ErrorFunction.GetError(y, target);
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
            if (ErrorFunction == null)
            {
                throw new InvalidOperationException("Layer error function is not specified!");
            }

            return ErrorFunction.BackpropagateError(Outputs, targets);
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

        internal void Initialize(int batchSize, int seqLen)
        {
            BatchSize = batchSize;
            SeqLen = seqLen;

            Initialize();
        }
    }
}