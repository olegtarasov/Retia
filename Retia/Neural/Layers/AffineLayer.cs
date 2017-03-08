using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Retia.Contracts;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public enum AffineActivation
    {
        None,
        Sigmoid
    }

    public class AffineLayer<T> : NeuroLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly LinearLayer<T> _linearLayer;
        private readonly NeuroLayer<T> _activationLayer;
        

        public AffineLayer(int xSize, int ySize, AffineActivation activation) 
        {

            _linearLayer = new LinearLayer<T>(xSize, ySize);
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(int xSize, int ySize, AffineActivation activation, IMatrixInitializer<T> matrixInitializer)
        {
            _linearLayer = new LinearLayer<T>(xSize, ySize, matrixInitializer);
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(BinaryReader reader)
        {
            // TODO: Implement load
        }

        public AffineLayer(AffineLayer<T> other) : base(other)
        {
            _linearLayer = (LinearLayer<T>) other._linearLayer.Clone();
            _activationLayer = other._activationLayer.Clone();
        }

        public override int InputSize => _linearLayer.InputSize;
        public override int OutputSize => _activationLayer.OutputSize;
        public override int TotalParamCount => _linearLayer.TotalParamCount + _activationLayer.TotalParamCount;

        public override NeuroLayer<T> Clone()
        {
            return new AffineLayer<T>(this);
        }

        public override List<Matrix<T>> ErrorPropagate(List<Matrix<T>> targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override void Optimize(OptimizerBase<T> optimizer)
        {
            _linearLayer.Optimize(optimizer);
            _activationLayer.Optimize(optimizer);
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            Inputs.Add(input);
            var output = _activationLayer.Step(_linearLayer.Step(input, inTraining), inTraining);
            Outputs.Add(output);
            return output;
        }

        public override void ResetMemory()
        {
            _linearLayer.ResetMemory();
            _activationLayer.ResetMemory();
        }

        public override void ResetOptimizer()
        {
            _linearLayer.ResetOptimizer();
            _activationLayer.ResetOptimizer();
        }

        public override void InitSequence()
        {
            Inputs.Clear();
            Outputs.Clear();
            _linearLayer.InitSequence();
            _activationLayer.InitSequence();
        }

        public override void ClampGrads(float limit)
        {
            _linearLayer.ClampGrads(limit);
            _activationLayer.ClampGrads(limit);
        }

        protected override void Initialize()
        {
            _linearLayer.Initialize(BatchSize, SeqLen);
        }

        public override LayerSpecBase CreateSpec()
        {
            throw new NotImplementedException();
        }

        public override void ToVectorState(T[] destination, ref int idx, bool grad = false)
        {
            _linearLayer.ToVectorState(destination, ref idx, grad);
            _activationLayer.ToVectorState(destination, ref idx, grad);
        }

        public override void FromVectorState(T[] vector, ref int idx)
        {
            _linearLayer.FromVectorState(vector, ref idx);
            _activationLayer.FromVectorState(vector, ref idx);
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true)
        {
            var activationSens = _activationLayer.BackPropagate(outSens);
            return _linearLayer.BackPropagate(activationSens, needInputSens);
        }

        private NeuroLayer<T> GetAffineActivationLayer(AffineActivation activation, int ySize)
        {
            switch (activation)
            {
                case AffineActivation.Sigmoid:
                    return new SigmoidLayer<T>(ySize);
                default:
                    throw new ArgumentOutOfRangeException(nameof(activation), activation, null);
            }
        }
    }
}