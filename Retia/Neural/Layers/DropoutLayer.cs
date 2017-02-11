using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Mathematics;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class DropoutLayer : NeuroLayer
    {
        private readonly int _size;
        private readonly float _dropout = 0.5f;

        private readonly List<Matrix> _masks = new List<Matrix>();
        private Matrix _scale;

        public DropoutLayer(int size, float dropout = 0.5f)
        {
            _size = size;
            _dropout = dropout;
        }

        public DropoutLayer(BinaryReader reader)
        {
            _size = reader.ReadInt32();
            _dropout = reader.ReadSingle();
        }

        private DropoutLayer()
        {
        }

        private DropoutLayer(DropoutLayer other) : base(other)
        {
            _size = other._size;
            _dropout = other._dropout;
            _masks = other._masks;
            _scale = other._scale;
        }

        public override int InputSize => _size;
        public override int OutputSize => _size;
        public override int TotalParamCount => 0;

        private int MaskSize => _masks.Count == 0 ? 0 : _masks[0].RowCount * _masks[0].ColumnCount;

        public override void Save(Stream s)
        {
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_size);
                writer.Write(_dropout);
            }
        }

        public override NeuroLayer Clone()
        {
            return new DropoutLayer(this);
        }

        public override void Optimize(OptimizerBase optimizer)
        {
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            if (input.ColumnCount != BatchSize)
                throw new InvalidOperationException($"Input has invalid batch size! Expected: {BatchSize}, got: {input.ColumnCount}");
            if (_size != input.RowCount)
                throw new Exception($"Wrong input matrix row size provided!\nExpected: {_size}, got: {input.RowCount}");

            if (inTraining)
            {
                var mask = MatrixFactory.RandomMaskMatrix(_size, BatchSize, _dropout);
                _masks.Add(mask);
                return (Matrix)input.PointwiseMultiply(mask);
            }

            //if not in training we scale all input activations by factor of dropout probability
            return (Matrix)input.PointwiseMultiply(_scale);
        }

        public override List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true)
        {
            if (_masks.Count == 0)
                throw new Exception("Empty step history, nothing to propagate!");
            if (outSens.Count != _masks.Count)
                throw new Exception("Not enough sensitivies in list!");

            //TODO: add more param checks (out sens dimensions == input x batchSize, etc)

            var seqLen = _masks.Count;
            var inputSensList = new List<Matrix>(seqLen);

            for (int i = seqLen - 1; i >= 0; i--)
            {
                var mask = _masks[i];
                var dOut = outSens[i];
                //this layer can not be first, so we always calculate input sens, disregarding needInputSens param
                var dInput = (Matrix)dOut.PointwiseMultiply(mask);
                inputSensList.Insert(0, dInput);
            }
            return inputSensList;
        }

        public override void ResetMemory()
        {
        }

        public override void ResetOptimizer()
        {
        }

        public override void InitSequence()
        {
            _masks.Clear();
        }

        public override void ClampGrads(float limit)
        {
        }

        public override void ToVectorState(double[] destination, ref int idx, bool grad = false)
        {
            //masks are dynamically generated during training, not needed in gen alg

            /*
            int maskSize = MaskSize;

            if (maskSize == 0)
            {
                return;
            }

            for (int i = 0; i < _masks.Count; i++)
            {
                _masks[i].CopyToArray(destination, ref idx);
            }*/
        }

        public override void FromVectorState(double[] vector, ref int idx)
        {
            /*
            int maskSize = MaskSize;
            if (maskSize == 0 || vector.Length == 0)
            {
                return;
            }

            int maskCount = vector.Length / MaskSize;
            for (int i = 0; i < maskCount; i++)
            {
                _masks[i].CopyFromArray(vector, ref idx);
            }*/
        }

        public override LayerSpecBase CreateSpec()
        {
            throw new NotSupportedException();
        }

        protected override void Initialize()
        {
            _scale = DenseMatrix.Create(_size, BatchSize, _dropout);
        }
    }
}