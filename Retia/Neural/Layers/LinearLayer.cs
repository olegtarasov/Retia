using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural.ErrorFunctions;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class LinearLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private NeuroWeight<T> _bias;
        private NeuroWeight<T> _weights;

        private LinearLayer(LinearLayer<T> other) : base(other)
        {
            _weights = other._weights.Clone();
            _bias = other._bias.Clone();

            RegisterWeights(_bias, _weights);
        }

        public LinearLayer(int xSize, int ySize) : this(xSize, ySize, new RandomMatrixInitializer<T>())
        {
        }

        public LinearLayer(int xSize, int ySize, IMatrixInitializer<T> matrixInitializer)
        {
            _weights = matrixInitializer.CreateMatrix(ySize, xSize);
            _bias = matrixInitializer.CreateMatrix(ySize, 1);

            ErrorFunction = new MeanSquareError<T>();

            RegisterWeights(_bias, _weights);
        }

        public LinearLayer(BinaryReader reader) : base(reader)
        {
            _bias = NeuroWeight<T>.Load(reader.BaseStream);
            _weights = NeuroWeight<T>.Load(reader.BaseStream);

            RegisterWeights(_bias, _weights);
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
            T min = MathProvider.Scalar(-limit);
            T max = MathProvider.Scalar(limit);

            _bias.Gradient.Clamp(min, max);
            _weights.Gradient.Clamp(min, max);
        }

        public override void ToVectorState(T[] destination, ref int idx, bool grad = false)
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

        public override void FromVectorState(T[] vector, ref int idx)
        {
            _bias.Weight.CopyFromArray(vector, ref idx);
            _weights.Weight.CopyFromArray(vector, ref idx);
        }

        public override void Save(Stream s)
        {
            base.Save(s);

            _bias.Save(s);
            _weights.Save(s);
        }

        public override LayerBase<T> Clone()
        {
            return new LinearLayer<T>(this);
        }

        public override void Optimize(OptimizerBase<T> optimizer)
        {
            optimizer.Optimize(_weights);
            optimizer.Optimize(_bias);
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            if (input.RowCount != _weights.Weight.ColumnCount)
                throw new Exception($"Wrong input matrix row size provided!\nExpected: {_weights.Weight.ColumnCount}, got: {input.RowCount}");
            if (input.ColumnCount != BatchSize)
                throw new Exception($"Wrong input batch size!\nExpected: {BatchSize}, got: {input.ColumnCount}");

            var output = _bias.Weight.TileColumns(input.ColumnCount);
            output.Accumulate(_weights.Weight, input);
            if (inTraining)
            {
                Inputs.Add(input);
                Outputs.Add(output);
            }
            return output;
        }

        public override List<Matrix<T>> ErrorPropagate(List<Matrix<T>> targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true, bool clearGrad = true)
        {
            if (clearGrad)
            {
                ClearGradients();
            }

            if (Inputs.Count == 0)
                throw new Exception("Empty inputs history, nothing to propagate!");
            if (outSens.Count != Inputs.Count)
                throw new Exception("Not enough sensitivies in list!");

            var yIdentity = Matrix<T>.Build.Dense(BatchSize, 1, Matrix<T>.One);
            var inputSensList = new List<Matrix<T>>(SeqLen);

            for (int i = SeqLen - 1; i >= 0; i--)
            {
                var sNext = outSens[i];
                var x = Inputs[i];
                _weights.Gradient.Accumulate(sNext, x, transposeB: Transpose.Transpose);
                if (BatchSize > 1)
                    _bias.Gradient.Accumulate(sNext, yIdentity);
                else
                    _bias.Gradient.Accumulate(sNext);
                if (needInputSens)
                {
                    var dInput = Matrix<T>.Build.Dense(x.RowCount, BatchSize);
                    dInput.Accumulate(_weights.Weight, sNext, Transpose.Transpose);
                    inputSensList.Insert(0, dInput);
                }
                else
                    inputSensList.Insert(0, Matrix<T>.Build.Dense(x.RowCount, BatchSize));
            }
            return inputSensList;
        }

        public override LayerSpecBase CreateSpec()
        {
            if (typeof(T) != typeof(float))
            {
                throw new InvalidOperationException("Only float for GPU!");
            }

            return new LinearLayerSpec(_weights.Weight.ColumnCount, BatchSize, SeqLen, _weights.Weight.RowCount, _weights.Weight as Matrix<float>, _bias.Weight as Matrix<float>);
        }

        public override void ClearGradients()
        {
            _weights.ClearGrad();
            _bias.ClearGrad();
        }
    }
}