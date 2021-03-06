﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using Retia.Interop;
using Retia.Mathematics;
using Retia.Neural.ErrorFunctions;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class GruLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly List<Matrix<T>> _hNewVals = new List<Matrix<T>>();
        private readonly List<Matrix<T>> _hPropVals = new List<Matrix<T>>();

        private readonly List<Matrix<T>> _rVals = new List<Matrix<T>>();
        private readonly List<Matrix<T>> _zVals = new List<Matrix<T>>();

        private readonly int _hSize;

        private NeuroWeight<T> _whh, _whr, _whz;
        private NeuroWeight<T> _wxh, _wxr, _wxz;
        private NeuroWeight<T> _bxh, _bxr, _bxz;
        private NeuroWeight<T> _bhh, _bhr, _bhz;
        private Matrix<T> _hiddenOnes;

        /// <summary>
        ///     Cache of last calculated hidden output.
        ///     Initially zero matrix
        /// </summary>
        private Matrix<T> _lastH;

        public GruLayer(int xSize, int hSize) : this(xSize, hSize, 
            new ProportionalRandomMatrixInitializer<T>(),
            new ProportionalRandomMatrixInitializer<T>(),
            new ConstantMatrixInitializer<T>())
        {
        }

        public GruLayer(int xSize, int hSize, 
            IMatrixInitializer<T> linearWeightInitializer,
            IMatrixInitializer<T> hiddenWeightInitializer,
            IMatrixInitializer<T> biasInitializer)
        {
            _hSize = hSize;

            _wxh = new NeuroWeight<T>(linearWeightInitializer.CreateMatrix(hSize, xSize));
            _wxr = new NeuroWeight<T>(linearWeightInitializer.CreateMatrix(hSize, xSize));
            _wxz = new NeuroWeight<T>(linearWeightInitializer.CreateMatrix(hSize, xSize));
                   
            _whh = new NeuroWeight<T>(hiddenWeightInitializer.CreateMatrix(hSize, hSize));
            _whr = new NeuroWeight<T>(hiddenWeightInitializer.CreateMatrix(hSize, hSize));
            _whz = new NeuroWeight<T>(hiddenWeightInitializer.CreateMatrix(hSize, hSize));
                   
            _bxh = new NeuroWeight<T>(biasInitializer.CreateMatrix(hSize, 1));
            _bxr = new NeuroWeight<T>(biasInitializer.CreateMatrix(hSize, 1));
            _bxz = new NeuroWeight<T>(biasInitializer.CreateMatrix(hSize, 1));
                   
            _bhh = new NeuroWeight<T>(biasInitializer.CreateMatrix(hSize, 1));
            _bhr = new NeuroWeight<T>(biasInitializer.CreateMatrix(hSize, 1));
            _bhz = new NeuroWeight<T>(biasInitializer.CreateMatrix(hSize, 1));

            ResetOptimizer();

            ErrorFunction = new MeanSquareError<T>();

            RegisterWeights();
        }

        public GruLayer(BinaryReader reader) : base(reader)
        {
            _bxr = NeuroWeight<T>.Load(reader.BaseStream);
            _bxz = NeuroWeight<T>.Load(reader.BaseStream);
            _bxh = NeuroWeight<T>.Load(reader.BaseStream);
                              
            _bhr = NeuroWeight<T>.Load(reader.BaseStream);
            _bhz = NeuroWeight<T>.Load(reader.BaseStream);
            _bhh = NeuroWeight<T>.Load(reader.BaseStream);
                              
            _wxr = NeuroWeight<T>.Load(reader.BaseStream);
            _wxz = NeuroWeight<T>.Load(reader.BaseStream);
            _wxh = NeuroWeight<T>.Load(reader.BaseStream);
                              
            _whr = NeuroWeight<T>.Load(reader.BaseStream);
            _whz = NeuroWeight<T>.Load(reader.BaseStream);
            _whh = NeuroWeight<T>.Load(reader.BaseStream);

            _lastH = MatrixFactory.Load<T>(reader.BaseStream);
            _hiddenOnes = Matrix<T>.Build.Dense(_hSize, _lastH.ColumnCount, Matrix<T>.One);

            RegisterWeights();
        }

        private GruLayer(GruLayer<T> other) : base(other)
        {
            _wxh = other._wxh.Clone();
            _wxr = other._wxr.Clone();
            _wxz = other._wxz.Clone();

            _whh = other._whh.Clone();
            _whr = other._whr.Clone();
            _whz = other._whz.Clone();

            _bxh = other._bxh.Clone();
            _bxr = other._bxr.Clone();
            _bxz = other._bxz.Clone();

            _bhh = other._bhh.Clone();
            _bhr = other._bhr.Clone();
            _bhz = other._bhz.Clone();

            _lastH = other._lastH.CloneMatrix();
            _hiddenOnes = other._hiddenOnes.CloneMatrix();

            Inputs = other.Inputs.Clone();
            Outputs = other.Outputs.Clone();
            _hPropVals = other._hPropVals.Clone();
            _hNewVals = other._hNewVals.Clone();
            _rVals = other._rVals.Clone();
            _zVals = other._zVals.Clone();
            _hSize = other._hSize;

            RegisterWeights();
        }

        public Matrix<T> HiddenState
        {
            get { return _lastH; }
            set
            {
                if (value == null) throw new ArgumentNullException(nameof(value));
                if (value.RowCount != _lastH.RowCount || value.ColumnCount != _lastH.ColumnCount)
                    throw new ArgumentOutOfRangeException(nameof(value), "Matrix dimensions mismatch!");

                _lastH = value;
            }
        }

        public override int InputSize => _wxh.Weight.ColumnCount;
        public override int OutputSize => _whh.Weight.RowCount;

        public override int TotalParamCount => _wxr.Weight.Length() + _wxz.Weight.Length() + _wxh.Weight.Length() +
                                               _whr.Weight.Length() + _whz.Weight.Length() + _whh.Weight.Length() +
                                               _bxr.Weight.Length() + _bxz.Weight.Length() + _bxh.Weight.Length() + 
                                               _bhr.Weight.Length() + _bhz.Weight.Length() + _bhh.Weight.Length();

        public override void Save(Stream stream)
        {
            base.Save(stream);

            _bxr.Save(stream);
            _bxz.Save(stream);
            _bxh.Save(stream);

            _bhr.Save(stream);
            _bhz.Save(stream);
            _bhh.Save(stream);

            _wxr.Save(stream);
            _wxz.Save(stream);
            _wxh.Save(stream);

            _whr.Save(stream);
            _whz.Save(stream);
            _whh.Save(stream);

            _lastH.Save(stream);
        }


        public override LayerBase<T> Clone()
        {
            return new GruLayer<T>(this);
        }

        public override void Optimize(OptimizerBase<T> optimizer)
        {
            optimizer.Optimize(_bxr);
            optimizer.Optimize(_bxz);
            optimizer.Optimize(_bxh);

            optimizer.Optimize(_bhr);
            optimizer.Optimize(_bhz);
            optimizer.Optimize(_bhh);

            optimizer.Optimize(_wxr);
            optimizer.Optimize(_wxz);
            optimizer.Optimize(_wxh);

            optimizer.Optimize(_whr);
            optimizer.Optimize(_whz);
            optimizer.Optimize(_whh);
        }

        public override List<Matrix<T>> ErrorPropagate(List<Matrix<T>> targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override void ResetMemory()
        {
            _lastH = Matrix<T>.Build.Dense(_lastH.RowCount, _lastH.ColumnCount);
        }

        public override void ResetOptimizer()
        {
            _bxr.ClearCache();
            _bxz.ClearCache();
            _bxh.ClearCache();

            _bhr.ClearCache();
            _bhz.ClearCache();
            _bhh.ClearCache();

            _wxr.ClearCache();
            _wxz.ClearCache();
            _wxh.ClearCache();

            _whr.ClearCache();
            _whz.ClearCache();
            _whh.ClearCache();
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            if (input.RowCount != _wxh.Weight.ColumnCount)
                throw new Exception($"Wrong input matrix row size provided!\nExpected: {_wxh.Weight.ColumnCount}, got: {input.RowCount}");
            if (input.ColumnCount != BatchSize)
                throw new Exception($"Wrong input batch size!\nExpected: {BatchSize}, got: {input.ColumnCount}");

            //var z = Bz + Wxz*input + Whz*lastH;
            var z = (_bxz.Weight.TileColumns(BatchSize) + _bhz.Weight.TileColumns(BatchSize));
            z.Accumulate(_wxz.Weight, input);
            z.Accumulate(_whz.Weight, _lastH);


            //var r = Br + Wxr*input + Whr*lastH;
            var r = (_bxr.Weight.TileColumns(BatchSize) + _bhr.Weight.TileColumns(BatchSize));
            r.Accumulate(_wxr.Weight, input);
            r.Accumulate(_whr.Weight, _lastH);

            //Sigmoid(z);
            //Sigmoid(r);
            //ActivationFuncs.ApplySigmoid(r);
            //ActivationFuncs.ApplySigmoid(z);
            MathProvider.ApplySigmoid2(r, z);
            //ApplySigmoid(r, z);


            var hNew = _bxh.Weight.TileColumns(BatchSize);
            hNew.Accumulate(_wxh.Weight, input);

            var hProp = _bhh.Weight.TileColumns(BatchSize);
            hProp.Accumulate(_whh.Weight, _lastH);
            
            hNew = hNew + r.PointwiseMultiply(hProp);
            MathProvider.ApplyTanh(hNew);
            //ApplyTanh(hNew);

            //var H = (z ^ hNew) + ((_hiddenOnes - z) ^ _lastH);
            var H = Matrix<T>.Build.Dense(hNew.RowCount, hNew.ColumnCount);
            MathProvider.CalculateH(H, hNew, z, _lastH);

            if (inTraining)
            {
                Outputs.Add(H);
                _hPropVals.Add(hProp);
                _hNewVals.Add(hNew);
                _zVals.Add(z);
                _rVals.Add(r);
                Inputs.Add(input);
            }

            _lastH = H;
            return H;
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true, bool clearGrad = true)
        {
            if (clearGrad)
            {
                ClearGradients();
            }

            var dh = Enumerable.Range(0, outSens.Count).Select(x => (Matrix<T>)null).ToList();
            var di = Enumerable.Range(0, outSens.Count).Select(x => (Matrix<T>)null).ToList();

            var batchOnes = Matrix<T>.Build.Dense(BatchSize, 1, Matrix<T>.One);

            for (int i = outSens.Count - 1; i >= 0; i--)
            {
                var dY = outSens[i];
                var dhNext = i == outSens.Count - 1 ? Matrix<T>.Build.Dense(_hSize, BatchSize) : dh[i + 1];
                var z = _zVals[i];
                var r = _rVals[i];
                var input = Inputs[i];
                var hProp = _hPropVals[i];
                var hPrev = i > 0 ? Outputs[i - 1] : Matrix<T>.Build.Dense(_hSize, BatchSize);
                var dhSum = (dY + dhNext);
                var hNew = _hNewVals[i];

                // bxh, Wxh
                var dbxh = dhSum.PointwiseMultiply(_hiddenOnes - z).PointwiseMultiply(_hiddenOnes - hNew.PointwiseMultiply(hNew)); // h x b
                _bxh.Gradient.CollapseColumnsAndAccumulate(dbxh, batchOnes); // h x 1
                _wxh.Gradient.Accumulate(dbxh, input, transposeB: Transpose.Transpose); // h x i

                // bhh, Whh
                var dbhh = dbxh.PointwiseMultiply(r); // h x b
                _bhh.Gradient.CollapseColumnsAndAccumulate(dbhh, batchOnes); // h x 1
                _whh.Gradient.Accumulate(dbhh, hPrev, transposeB: Transpose.Transpose); // h x h

                // bxr, Wxr
                var dbxr = dbxh.PointwiseMultiply(hProp).PointwiseMultiply(r.PointwiseMultiply(_hiddenOnes - r)); // h x b
                _bxr.Gradient.CollapseColumnsAndAccumulate(dbxr, batchOnes); // h x 1
                _wxr.Gradient.Accumulate(dbxr, input, transposeB: Transpose.Transpose); // h x i

                // bhr, whr
                var dbhr = dbxr; // h x b
                _bhr.Gradient.CollapseColumnsAndAccumulate(dbhr, batchOnes); // h x 1
                _whr.Gradient.Accumulate(dbhr, hPrev, transposeB: Transpose.Transpose); // h x h

                // bxz, wxz
                var dbxz = dhSum.PointwiseMultiply(hPrev - hNew).PointwiseMultiply(z.PointwiseMultiply(_hiddenOnes - z)); // h x b
                _bxz.Gradient.CollapseColumnsAndAccumulate(dbxz, batchOnes); // h x 1
                _wxz.Gradient.Accumulate(dbxz, input, transposeB: Transpose.Transpose); // h x i

                // bhz, whz
                var dbhz = dbxz; // h x b
                _bhz.Gradient.CollapseColumnsAndAccumulate(dbhz, batchOnes); // h x 1
                _whz.Gradient.Accumulate(dbhz, hPrev, transposeB: Transpose.Transpose); // h x h

                dh[i] = dhSum.PointwiseMultiply(z) + _whz.Weight.Transpose() * dbhz + _whr.Weight.Transpose() * dbhr + _whh.Weight.Transpose() * dbhh;

                if (needInputSens)
                {
                    di[i] = _wxz.Weight.Transpose() * dbxz + _wxr.Weight.Transpose() * dbxr + _wxh.Weight.Transpose() * dbxh;
                }
            }

            return di;
        }

        public override void ClampGrads(float limit)
        {
            T min = MathProvider.Scalar(-limit);
            T max = MathProvider.Scalar(limit);

            _whr.Gradient.Clamp(min, max);
            _whz.Gradient.Clamp(min, max);
            _whh.Gradient.Clamp(min, max);
                                
            _wxr.Gradient.Clamp(min, max);
            _wxz.Gradient.Clamp(min, max);
            _wxh.Gradient.Clamp(min, max);
                                
            _bxr.Gradient.Clamp(min, max);
            _bxz.Gradient.Clamp(min, max);
            _bxh.Gradient.Clamp(min, max);
                                
            _bhr.Gradient.Clamp(min, max);
            _bhz.Gradient.Clamp(min, max);
            _bhh.Gradient.Clamp(min, max);
        }

        public override void ToVectorState(T[] destination, ref int idx, bool grad = false)
        {
            if (!grad)
            {
                _bxr.Weight.CopyToArray(destination, ref idx);
                _bxz.Weight.CopyToArray(destination, ref idx);
                _bxh.Weight.CopyToArray(destination, ref idx);

                _bhr.Weight.CopyToArray(destination, ref idx);
                _bhz.Weight.CopyToArray(destination, ref idx);
                _bhh.Weight.CopyToArray(destination, ref idx);

                _wxr.Weight.CopyToArray(destination, ref idx);
                _wxz.Weight.CopyToArray(destination, ref idx);
                _wxh.Weight.CopyToArray(destination, ref idx);

                _whr.Weight.CopyToArray(destination, ref idx);
                _whz.Weight.CopyToArray(destination, ref idx);
                _whh.Weight.CopyToArray(destination, ref idx);
            }
            else
            {
                _bxr.Gradient.CopyToArray(destination, ref idx);
                _bxz.Gradient.CopyToArray(destination, ref idx);
                _bxh.Gradient.CopyToArray(destination, ref idx);

                _bhr.Gradient.CopyToArray(destination, ref idx);
                _bhz.Gradient.CopyToArray(destination, ref idx);
                _bhh.Gradient.CopyToArray(destination, ref idx);

                _wxr.Gradient.CopyToArray(destination, ref idx);
                _wxz.Gradient.CopyToArray(destination, ref idx);
                _wxh.Gradient.CopyToArray(destination, ref idx);

                _whr.Gradient.CopyToArray(destination, ref idx);
                _whz.Gradient.CopyToArray(destination, ref idx);
                _whh.Gradient.CopyToArray(destination, ref idx);
            }
        }

        public override void FromVectorState(T[] vector, ref int idx)
        {
            _bxr.Weight.CopyFromArray(vector, ref idx);
            _bxz.Weight.CopyFromArray(vector, ref idx);
            _bxh.Weight.CopyFromArray(vector, ref idx);

            _bhr.Weight.CopyFromArray(vector, ref idx);
            _bhz.Weight.CopyFromArray(vector, ref idx);
            _bhh.Weight.CopyFromArray(vector, ref idx);

            _wxr.Weight.CopyFromArray(vector, ref idx);
            _wxz.Weight.CopyFromArray(vector, ref idx);
            _wxh.Weight.CopyFromArray(vector, ref idx);

            _whr.Weight.CopyFromArray(vector, ref idx);
            _whz.Weight.CopyFromArray(vector, ref idx);
            _whh.Weight.CopyFromArray(vector, ref idx);
        }


        public override void InitSequence()
        {
            Outputs.Clear();
            _hPropVals.Clear();
            _hNewVals.Clear();
            _zVals.Clear();
            _rVals.Clear();
            Inputs.Clear();
            ResetMemory();
        }

        protected override void Initialize()
        {
            _lastH = Matrix<T>.Build.Dense(_hSize, BatchSize);
            _hiddenOnes = Matrix<T>.Build.Dense(_hSize, BatchSize, Matrix<T>.One);
            InitSequence();
        }

        private void RegisterWeights()
        {
            RegisterWeights(_wxr,
                _wxz,
                _wxh,

                _whr,
                _whz,
                _whh,

                _bxr,
                _bxz,
                _bxh,

                _bhr,
                _bhz,
                _bhh);
        }

        public override void ClearGradients()
        {
            _bxr.ClearGrad();
            _bxz.ClearGrad();
            _bxh.ClearGrad();

            _bhr.ClearGrad();
            _bhz.ClearGrad();
            _bhh.ClearGrad();

            _wxr.ClearGrad();
            _wxz.ClearGrad();
            _wxh.ClearGrad();

            _whr.ClearGrad();
            _whz.ClearGrad();
            _whh.ClearGrad();
        }

        public override IntPtr CreateGpuLayer()
        {
            GpuLayerPtr = GpuInterface.CreateGruLayer(InputSize, _hSize, 1, BatchSize, SeqLen);
            TransferWeightsToDevice();

            return GpuLayerPtr;
        }

        public override void TransferWeightsToDevice()
        {
            TransferWeigthsToDevice(true, // CuDNN weight matrices are row-major
                _wxr,
                _wxz,
                _wxh,

                _whr,
                _whz,
                _whh,

                _bxr,
                _bxz,
                _bxh,

                _bhr,
                _bhz,
                _bhh);
        }

        public override void TransferWeightsToHost()
        {
            TransferWeigthsToHost(true, // CuDNN weight matrices are row-major
                _wxr,
                _wxz,
                _wxh,

                _whr,
                _whz,
                _whh,

                _bxr,
                _bxz,
                _bxh,

                _bhr,
                _bhz,
                _bhh);
        }
    }
}