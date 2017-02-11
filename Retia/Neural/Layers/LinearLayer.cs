using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Providers.LinearAlgebra;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class LinearLayer : NeuroLayer
    {
        private const double Dispersion = 5e-2;

        public NeuroWeight _bias;
        public NeuroWeight _weights;

        private LinearLayer(LinearLayer other) : base(other)
        {
            _weights = other._weights.Clone();
            _bias = other._bias.Clone();
        }

        private LinearLayer()
        {
        }

        public LinearLayer(int xSize, int ySize) : this(xSize, ySize, new RandomMatrixInitializer {Dispersion = 5})
        {
            
        }

        public LinearLayer(int xSize, int ySize, IMatrixInitializer matrixInitializer)
        {
            _weights = matrixInitializer.CreateMatrix(ySize, xSize);
            _bias = matrixInitializer.CreateMatrix(ySize, 1);
        }

        public LinearLayer(BinaryReader reader)
        {
            _bias = NeuroWeight.Load(reader.BaseStream);
            _weights = NeuroWeight.Load(reader.BaseStream);
        }

        public override int InputSize => _weights.Weight.ColumnCount;
        public override int OutputSize => _weights.Weight.RowCount;
        public override int TotalParamCount => _weights.Weight.AsColumnMajorArray().Length + _bias.Weight.AsColumnMajorArray().Length;

        public override void InitSequence()
        {
            Inputs.Clear();
            Outputs.Clear();
        }

        public override void ResetMemory()
        {
            //nothing to do here
        }

        public override void ResetOptimizer()
        {
            _bias.ClearCache();
            _weights.ClearCache();
        }

        public override void ClampGrads(float limit)
        {
            _bias.Gradient.Clamp(-limit, limit);
            _weights.Gradient.Clamp(-limit, limit);
        }

        public override void ToVectorState(double[] destination, ref int idx, bool grad = false)
        {
            if (!grad)
            {
                _bias.Weight.CopyToArray(destination, ref idx);
                _weights.Weight.CopyToArray(destination, ref idx);
            }
            else
            {
                _bias.Gradient.CopyToArray(destination, ref idx);
                _weights.Gradient.CopyToArray(destination, ref idx);
            }
        }

        public override void FromVectorState(double[] vector, ref int idx)
        {
            _bias.Weight.CopyFromArray(vector, ref idx);
            _weights.Weight.CopyFromArray(vector, ref idx);
        }

        public override double LayerError(Matrix y, Matrix target)
        {
            return ErrorFunctions.MeanSquare(y, target);
        }

        public override void Save(Stream s)
        {
            _bias.Save(s);
            _weights.Save(s);
        }

        public override NeuroLayer Clone()
        {
            return new LinearLayer(this);
        }

        public override void Optimize(OptimizerBase optimizer)
        {
            optimizer.Optimize(_weights);
            optimizer.Optimize(_bias);
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            if (input.RowCount != _weights.Weight.ColumnCount)
                throw new Exception($"Wrong input matrix row size provided!\nExpected: {_weights.Weight.ColumnCount}, got: {input.RowCount}");
            if (input.ColumnCount != BatchSize)
                throw new Exception($"Wrong input batch size!\nExpected: {BatchSize}, got: {input.ColumnCount}");

            var output = _bias.Weight.TileColumns(input.ColumnCount);
            output.Accumulate(_weights.Weight, input, 1.0f);
            if (inTraining)
            {
                Inputs.Add(input);
                Outputs.Add(output);
            }
            return output;
        }

        public override List<Matrix> ErrorPropagate(List<Matrix> targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true)
        {
            _weights.ClearGrad();
            _bias.ClearGrad();

            if (Inputs.Count == 0)
                throw new Exception("Empty inputs history, nothing to propagate!");
            if (outSens.Count != Inputs.Count)
                throw new Exception("Not enough sensitivies in list!");

            var yIdentity = DenseMatrix.Create(BatchSize, 1, DenseMatrix.One);
            var inputSensList = new List<Matrix>(SeqLen);

            for (int i = SeqLen - 1; i >= 0; i--)
            {
                var sNext = outSens[i];
                var x = Inputs[i];
                _weights.Gradient.Accumulate(sNext, x, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
                if (BatchSize > 1)
                    _bias.Gradient.Accumulate(sNext, yIdentity, 1.0f);
                else
                    _bias.Gradient.Accumulate(sNext);
                if (needInputSens)
                {
                    var dInput = new DenseMatrix(x.RowCount, BatchSize);
                    dInput.Accumulate(_weights.Weight, sNext, 1.0f, 1.0f, Transpose.Transpose);
                    inputSensList.Insert(0, dInput);
                }
                else
                    inputSensList.Insert(0, new DenseMatrix(x.RowCount, BatchSize));
            }
            return inputSensList;
        }

        public override LayerSpecBase CreateSpec()
        {
            return null;// new LinearLayerSpec(_weights.Weight.ColumnCount, BatchSize, SeqLen, _weights.Weight.RowCount, _weights.Weight, _bias.Weight);
        }

        //we need to propagate matched error through weight matrix
    }
}