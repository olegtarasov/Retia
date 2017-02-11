using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public abstract class NeuroLayer : ICloneable<NeuroLayer>, IFileWritable
    {
        protected int BatchSize;
        protected int SeqLen;

        protected List<Matrix> Inputs = new List<Matrix>();
        protected List<Matrix> Outputs = new List<Matrix>();

        protected NeuroLayer()
        {
        }

        protected NeuroLayer(NeuroLayer other)
        {
            BatchSize = other.BatchSize;
            SeqLen = other.SeqLen;
            Inputs = other.Inputs.Select(x => x.CloneMatrix()).ToList();
            Outputs = other.Outputs.Select(x => x.CloneMatrix()).ToList();
        }

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }
        public abstract int TotalParamCount { get; }
        public virtual Matrix[] InternalState { get;  set; }

        public abstract NeuroLayer Clone();

        public abstract void Save(Stream s);

        public abstract void Optimize(OptimizerBase optimizer);

        /// <summary>
        ///     Forward layer step
        /// </summary>
        /// <param name="input">Input matrix</param>
        /// <param name="inTraining">Store states for back propagation</param>
        /// <returns>Layer output</returns>
        public abstract Matrix Step(Matrix input, bool inTraining = false);

        public abstract void ResetMemory();
        public abstract void ResetOptimizer();
        public abstract void InitSequence();
        public abstract void ClampGrads(float limit);
        public abstract LayerSpecBase CreateSpec();

        /// <summary>
        /// Converts layer state to a vector of doubles.
        /// </summary>
        /// <returns>Layer vector state.</returns>
        public abstract void ToVectorState(double[] destination, ref int idx, bool grad = false);

        /// <summary>
        /// Modifies layer state from a vector of doubles.
        /// </summary>
        public abstract void FromVectorState(double[] vector, ref int idx);


        /// <summary>
        ///     Propagates next layer sensitivity to input, accumulating gradients for optimization
        /// </summary>
        /// <param name="outSens">Sequence of sensitivity matrices of next layer</param>
        /// <param name="needInputSens">Calculate input sensitivity for further propagation</param>
        /// <returns></returns>
        public virtual List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true)
        {
            return outSens;
        }


        /// <summary>
        ///     Propagates next layer sensitivity to input, calculating input sensitivity matrix
        /// </summary>
        /// <param name="outSens">Sequence of sensitivity matrices of next layer</param>
        /// <returns></returns>
        public virtual List<Matrix> PropagateSensitivity(List<Matrix> outSens)
        {
            var iSensList = new List<Matrix>(Inputs.Count);
            for (int step = 0; step < Inputs.Count; step++)
            {
                var oSens = outSens[step];
                var iSens = new DenseMatrix(InputSize, BatchSize);
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
        public virtual double LayerError(Matrix y, Matrix target)
        {
            return 0.0;
        }

        public virtual void SetParam(int i, double value)
        {
            int refInd = 0;
            var state = new double[TotalParamCount];
            ToVectorState(state, ref refInd);
            state[i] = value;
            refInd = 0;
            FromVectorState(state, ref refInd);
        }

        /// <summary>
        ///     Calculates matched error (out-target) and propagates it through layer to inputs
        /// </summary>
        /// <param name="targets">Sequence of targets</param>
        public virtual List<Matrix> ErrorPropagate(List<Matrix> targets)
        {
            if (Outputs.Count != targets.Count || targets.Count == 0)
                throw new Exception("Not enough targets provided or not enough output states stored!");

            var sensitivities = new List<Matrix>(SeqLen);
            float k = 1.0f / BatchSize;

            for (int i = 0; i < SeqLen; i++)
            {
                var y = Outputs[i];
                var target = targets[i];
                var sensitivity = (Matrix)new DenseMatrix(y.RowCount, y.ColumnCount);

                var ya = y.AsColumnMajorArray();
                var ta = target.AsColumnMajorArray();
                var sa = sensitivity.AsColumnMajorArray();

                for (int idx = 0; idx < ya.Length; idx++)
                {
                    double t = ta[idx];
                    sa[idx] = double.IsNaN(t) ? 0.0f : (ya[idx] - t) * k;
                }

                sensitivities.Add(sensitivity);
            }

            return sensitivities;
        }

        public void Save(string filename)
        {
            this.SaveObject(filename);
        }

        public double GetParam(int i, bool grad = false)
        {
            int refInd=0;
            var state = new double[TotalParamCount];
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
        protected virtual double Derivative(Matrix input, Matrix output, int batch, int i, int o)
        {
            return 0.0f;
        }

        internal void Initialize(int batchSize, int seqLen)
        {
            BatchSize = batchSize;
            SeqLen = seqLen;

            Initialize();
        }

        private void CalcSens(int step, int batch, Matrix iSens, Matrix outSens)
        {
            for (int i = 0; i < InputSize; i++)
            {
                for (int o = 0; o < OutputSize; o++)
                {
                    iSens[i, batch] += Derivative(Inputs[step], Outputs[step], batch, i, o) * outSens[o, batch];
                }
            }
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