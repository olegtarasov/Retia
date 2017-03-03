using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Retia.Neural.Initializers;

namespace Retia.Neural.Layers
{
    public enum AffineActivation
    {
        None,
        Sigmoid
    }

    public class AffineLayer<T> : LinearLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly NeuroLayer<T> _activationLayer;

        public AffineLayer(int xSize, int ySize, AffineActivation activation) : base(xSize, ySize)
        {
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(int xSize, int ySize, AffineActivation activation, IMatrixInitializer<T> matrixInitializer) : base(xSize, ySize, matrixInitializer)
        {
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(BinaryReader reader) : base(reader)
        {
            // TODO: Implement load
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            return _activationLayer.Step(base.Step(input, inTraining), true);
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true)
        {
            var activationSens = _activationLayer.BackPropagate(outSens);
            return base.BackPropagate(activationSens, needInputSens);
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