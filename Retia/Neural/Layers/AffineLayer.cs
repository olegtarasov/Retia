using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public enum AffineActivation
    {
        None,
        Sigmoid,
        Tanh
    }

    public class AffineLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly LayerBase<T> _activationLayer;
        private readonly AffineActivation _activationType;
        private readonly LinearLayer<T> _linearLayer;


        public AffineLayer(int xSize, int ySize, AffineActivation activation)
        {
            _activationType = activation;
            _linearLayer = new LinearLayer<T>(xSize, ySize);
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(int xSize, int ySize, AffineActivation activation, IMatrixInitializer<T> matrixInitializer)
        {
            _activationType = activation;
            _linearLayer = new LinearLayer<T>(xSize, ySize, matrixInitializer);
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(BinaryReader reader)
        {
            var activationType = (AffineActivation)reader.ReadInt32();
            if (activationType == AffineActivation.None)
            {
                throw new InvalidOperationException("Invalid activation type loaded from file!");
            }

            _linearLayer = new LinearLayer<T>(reader);
            _activationLayer = LoadAffineActivationLayer(activationType, reader);
        }

        public AffineLayer(AffineLayer<T> other) : base(other)
        {
            _activationType = other._activationType;
            _linearLayer = (LinearLayer<T>)other._linearLayer.Clone();
            _activationLayer = other._activationLayer.Clone();
        }

        public override int InputSize => _linearLayer.InputSize;

        public override int OutputSize => _activationLayer.OutputSize;
        public override int TotalParamCount => _linearLayer.TotalParamCount + _activationLayer.TotalParamCount;

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true, bool clearGrad = true)
        {
            var activationSens = _activationLayer.BackPropagate(outSens, needInputSens, clearGrad);
            return _linearLayer.BackPropagate(activationSens, needInputSens, clearGrad);
        }

        public override void ClampGrads(float limit)
        {
            _linearLayer.ClampGrads(limit);
            _activationLayer.ClampGrads(limit);
        }

        public override LayerBase<T> Clone()
        {
            return new AffineLayer<T>(this);
        }

        public override List<Matrix<T>> ErrorPropagate(List<Matrix<T>> targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override void FromVectorState(T[] vector, ref int idx)
        {
            _linearLayer.FromVectorState(vector, ref idx);
            _activationLayer.FromVectorState(vector, ref idx);
        }

        public override void InitSequence()
        {
            Inputs.Clear();
            Outputs.Clear();
            _linearLayer.InitSequence();
            _activationLayer.InitSequence();
        }

        public override void Optimize(OptimizerBase<T> optimizer)
        {
            _linearLayer.Optimize(optimizer);
            _activationLayer.Optimize(optimizer);
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

        public override void Save(Stream s)
        {
            base.Save(s);
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write((int)_activationType);
                _linearLayer.Save(s);
                _activationLayer.Save(s);
            }
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            Inputs.Add(input);
            var output = _activationLayer.Step(_linearLayer.Step(input, inTraining), inTraining);
            Outputs.Add(output);
            return output;
        }

        public override void ToVectorState(T[] destination, ref int idx, bool grad = false)
        {
            _linearLayer.ToVectorState(destination, ref idx, grad);
            _activationLayer.ToVectorState(destination, ref idx, grad);
        }

        protected override void Initialize()
        {
            _linearLayer.Initialize(BatchSize, SeqLen);
            _activationLayer.Initialize(BatchSize, SeqLen);
        }

        private LayerBase<T> GetAffineActivationLayer(AffineActivation activation, int ySize)
        {
            switch (activation)
            {
                case AffineActivation.Sigmoid:
                    return new SigmoidLayer<T>(ySize);
                case AffineActivation.Tanh:
                    return new TanhLayer<T>(ySize);
                default:
                    throw new ArgumentOutOfRangeException(nameof(activation), activation, null);
            }
        }

        private LayerBase<T> LoadAffineActivationLayer(AffineActivation activation, BinaryReader reader)
        {
            switch (activation)
            {
                case AffineActivation.Sigmoid:
                    return new SigmoidLayer<T>(reader);
                case AffineActivation.Tanh:
                    return new TanhLayer<T>(reader);
                default:
                    throw new ArgumentOutOfRangeException(nameof(activation), activation, null);
            }
        }

        public override IReadOnlyList<NeuroWeight<T>> Weights => _linearLayer.Weights;
        public override void ClearGradients()
        {
            _linearLayer.ClearGradients();
            _activationLayer.ClearGradients();
        }

        public override IntPtr CreateGpuLayer()
        {
            throw new NotSupportedException();
        }
    }
}