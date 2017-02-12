using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class GruLayer<T> : NeuroLayer<T> where T : struct, IEquatable<T>, IFormattable
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

        public override Matrix<T>[] InternalState
        {
            get { return new []{_lastH.CloneMatrix()}; }
            set
            {
                if(value.Count()!=1)
                    throw new Exception($"Internal state of {GetType().AssemblyQualifiedName} should consist of 1 matrix");
                _lastH = value[0];
            }
        }

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

            _wxh = linearWeightInitializer.CreateMatrix(hSize, xSize);
            _wxr = linearWeightInitializer.CreateMatrix(hSize, xSize);
            _wxz = linearWeightInitializer.CreateMatrix(hSize, xSize);

            _whh = hiddenWeightInitializer.CreateMatrix(hSize, hSize);
            _whr = hiddenWeightInitializer.CreateMatrix(hSize, hSize);
            _whz = hiddenWeightInitializer.CreateMatrix(hSize, hSize);

            _bxh = biasInitializer.CreateMatrix(hSize, 1);
            _bxr = biasInitializer.CreateMatrix(hSize, 1);
            _bxz = biasInitializer.CreateMatrix(hSize, 1);

            _bhh = biasInitializer.CreateMatrix(hSize, 1);
            _bhr = biasInitializer.CreateMatrix(hSize, 1);
            _bhz = biasInitializer.CreateMatrix(hSize, 1);

            ResetOptimizer();
        }

        public GruLayer(BinaryReader reader)
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
        }

        public override int InputSize => _wxh.Weight.ColumnCount;
        public override int OutputSize => _whh.Weight.RowCount;

        // TODO: Clean this shit up
        public override int TotalParamCount => _wxr.Weight.Length() + _wxz.Weight.Length() + _wxh.Weight.Length() +
                                               _whr.Weight.Length() + _whz.Weight.Length() + _whh.Weight.Length() +
                                               _bxr.Weight.Length() + _bxz.Weight.Length() + _bxh.Weight.Length() + 
                                               _bhr.Weight.Length() + _bhz.Weight.Length() + _bhh.Weight.Length();

        public override void Save(Stream stream)
        {
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


        public override NeuroLayer<T> Clone()
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

        protected override void Initialize()
        {
            _lastH = Matrix<T>.Build.Dense(_hSize, BatchSize);
            _hiddenOnes = Matrix<T>.Build.Dense(_hSize, BatchSize, Matrix<T>.One);
            InitSequence();
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            if (input.RowCount != _wxh.Weight.ColumnCount)
                throw new Exception($"Wrong input matrix row size provided!\nExpected: {_wxh.Weight.ColumnCount}, got: {input.RowCount}");
            if (input.ColumnCount != BatchSize)
                throw new Exception($"Wrong input batch size!\nExpected: {BatchSize}, got: {input.ColumnCount}");

            //var z = Bz + Wxz*input + Whz*lastH;
            var z = (_bxz.Weight.TileColumns(BatchSize) + _bhz.Weight.TileColumns(BatchSize));
            z.Accumulate(_wxz.Weight, input, 1.0f);
            z.Accumulate(_whz.Weight, _lastH, 1.0f);


            //var r = Br + Wxr*input + Whr*lastH;
            var r = (_bxr.Weight.TileColumns(BatchSize) + _bhr.Weight.TileColumns(BatchSize));
            r.Accumulate(_wxr.Weight, input, 1.0f);
            r.Accumulate(_whr.Weight, _lastH, 1.0f);

            //Sigmoid(z);
            //Sigmoid(r);
            //ActivationFuncs.ApplySigmoid(r);
            //ActivationFuncs.ApplySigmoid(z);
            MathProvider.ApplySigmoid2(r, z);
            //ApplySigmoid(r, z);


            var hNew = _bxh.Weight.TileColumns(BatchSize);
            hNew.Accumulate(_wxh.Weight, input, 1.0f);

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

        private List<Matrix<T>> BackpropNew(List<Matrix<T>> sensitivities, bool needInputSens)
        {
            var dh = Enumerable.Range(0, sensitivities.Count).Select(x => (Matrix<T>)null).ToList();
            var di = Enumerable.Range(0, sensitivities.Count).Select(x => (Matrix<T>)null).ToList();

            var batchOnes = Matrix<T>.Build.Dense(BatchSize, 1, Matrix<T>.One);

            for (int i = sensitivities.Count - 1; i >= 0; i--)
            {
                var dY = sensitivities[i];
                var dhNext = i == sensitivities.Count - 1 ? Matrix<T>.Build.Dense(_hSize, BatchSize) : dh[i + 1];
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
                _wxh.Gradient.Accumulate(dbxh, input, 1.0f, transposeB: Transpose.Transpose); // h x i

                // bhh, Whh
                var dbhh = dbxh.PointwiseMultiply(r); // h x b
                _bhh.Gradient.CollapseColumnsAndAccumulate(dbhh, batchOnes); // h x 1
                _whh.Gradient.Accumulate(dbhh, hPrev, 1.0f, transposeB: Transpose.Transpose); // h x h
                
                // bxr, Wxr
                var dbxr = dbxh.PointwiseMultiply(hProp).PointwiseMultiply(r.PointwiseMultiply(_hiddenOnes - r)); // h x b
                _bxr.Gradient.CollapseColumnsAndAccumulate(dbxr, batchOnes); // h x 1
                _wxr.Gradient.Accumulate(dbxr, input, 1.0f, transposeB: Transpose.Transpose); // h x i
                
                // bhr, whr
                var dbhr = dbxr; // h x b
                _bhr.Gradient.CollapseColumnsAndAccumulate(dbhr, batchOnes); // h x 1
                _whr.Gradient.Accumulate(dbhr, hPrev, 1.0f, transposeB: Transpose.Transpose); // h x h
                
                // bxz, wxz
                var dbxz = dhSum.PointwiseMultiply(hPrev - hNew).PointwiseMultiply(z.PointwiseMultiply(_hiddenOnes - z)); // h x b
                _bxz.Gradient.CollapseColumnsAndAccumulate(dbxz, batchOnes); // h x 1
                _wxz.Gradient.Accumulate(dbxz, input, 1.0f, transposeB: Transpose.Transpose); // h x i
                
                // bhz, whz
                var dbhz = dbxz; // h x b
                _bhz.Gradient.CollapseColumnsAndAccumulate(dbhz, batchOnes); // h x 1
                _whz.Gradient.Accumulate(dbhz, hPrev, 1.0f, transposeB: Transpose.Transpose); // h x h
                
                dh[i] = (dhSum.PointwiseMultiply(z) + (_whz.Weight.Transpose() * dbxz) + (_whr.Weight.Transpose() * dbxr) + (_whh.Weight.Transpose() * dbhh));

                if (needInputSens)
                {
                    di[i] = (_wxz.Weight.Transpose() * dbxz + _wxr.Weight.Transpose() * dbxr + _wxh.Weight.Transpose() * dbxh);
                }
            }

            return di;
        }

        //private void CalcBackHadamards(Matrix sH, Matrix sR, Matrix sZ, Matrix sHprop, Matrix sHnext, Matrix z,
        //                               Matrix r, Matrix h,
        //                               Matrix newH,
        //                               Matrix prevH, Matrix sY, Matrix propH)
        //{
        //    var _sH = sH.AsColumnMajorArray();
        //    var _sR = sR.AsColumnMajorArray();
        //    var _sZ = sZ.AsColumnMajorArray();
        //    var _sHnext = sHnext.AsColumnMajorArray();
        //    var _sHprop = sHprop.AsColumnMajorArray();
        //    var _z = z.AsColumnMajorArray();
        //    var _r = r.AsColumnMajorArray();
        //    var _h = h.AsColumnMajorArray();
        //    var _prevH = prevH.AsColumnMajorArray();
        //    var _sY = sY.AsColumnMajorArray();
        //    var _propH = propH.AsColumnMajorArray();
        //    var _newH = newH.AsColumnMajorArray();

        //    /*
        //    var sO = sHnext + sY;
        //    var derH = hiddenOnes - (newH ^ newH);
        //    var sH = derH ^ z ^ sO;
        //    var sHprop = sH ^ r;

        //    var derR = r ^ (hiddenOnes - r);
        //    var sR = derR ^ propH ^ sH;

        //    var derZ = z ^ (hiddenOnes - z);
        //    var sZ = derZ ^ (newH - prevH) ^ sO;

        //    //The way prevH influence current state
        //    sHnext = ((hiddenOnes - z) ^ sO);
        //    */

        //    for (var i = 0; i < _hiddenOnes.RowCount * _hiddenOnes.ColumnCount; i++)
        //    {
        //        var sO = _sHnext[i] + _sY[i];
        //        var derH = 1 - _newH[i] * _newH[i];
        //        _sH[i] = derH * _z[i] * sO;
        //        _sHprop[i] = _sH[i] * _r[i];
        //        var derR = _r[i] * (1.0f - _r[i]);
        //        _sR[i] = derR * _propH[i] * _sH[i];
        //        var derZ = _z[i] * (1.0f - _z[i]);
        //        _sZ[i] = derZ * (_newH[i] - _prevH[i]) * sO;
        //        _sHnext[i] = (1 - _z[i]) * sO;
        //    }
        //}

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true)
        {
            return BackpropNew(outSens, needInputSens);

            //if (Outputs.Count != Inputs.Count)
            //    throw new Exception("Backprop was not initialized (empty state sequence)");
            //if (Inputs.Count == 0)
            //    throw new Exception("Empty inputs history, nothing to propagate!");
            //if (outSens.Count != Inputs.Count)
            //    throw new Exception("Not enough sensitivies in list!");

            //var inputSensList = new List<Matrix>(SeqLen);

            //var hiddenIdentity = DenseMatrix.Create(BatchSize, 1, DenseMatrix.One);
            //var yIdentity = DenseMatrix.Create(BatchSize, 1, DenseMatrix.One);


            //_bxr.ClearGrad();
            //_bxz.ClearGrad();
            //_bxh.ClearGrad();

            //_bhr.ClearGrad();
            //_bhz.ClearGrad();
            //_bhh.ClearGrad();

            //_wxr.ClearGrad();
            //_wxz.ClearGrad();
            //_wxh.ClearGrad();

            //_whr.ClearGrad();
            //_whz.ClearGrad();
            //_whh.ClearGrad();


            ////Sensitivity of next state
            //var sHnext = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);

            //for (var i = Inputs.Count - 1; i >= 0; i--)
            //{
            //    //output sensitivity
            //    var sY = outSens[i];

            //    //inputs
            //    var x = Inputs[i];

            //    //R matrix
            //    var r = _rVals[i];

            //    //Z matrix
            //    var z = _zVals[i];

            //    //Previous hidden value

            //    var prevH = i > 0 ? Outputs[i - 1] : new DenseMatrix(OutputSize, BatchSize);

            //    //Current hidden value
            //    var H = Outputs[i];

            //    //Weighted previous hidden value
            //    var propH = _hPropVals[i];

            //    //Current hidden candidate
            //    var newH = _hNewVals[i];


            //    //Transponsed martices
            //    /*
            //    var tPrevH = Matrix.Transpose(prevH);
            //    var tX = Matrix.Transpose(x);
            //    */

            //    //Sigmoid derrivative:  f'(x) = f(x)*(1-f(x))
            //    //Tanh derrivative:     f'(x) = 1-f(x)*f(x)

            //    //var sO = sHnext + sY;


            //    /*
            //    var derH = hiddenOnes - (newH ^ newH);
            //    var sH = derH ^ z ^ sO;

            //    var sHprop = sH ^ r;

            //    var derR = r ^ (hiddenOnes - r);
            //    var sR = derR ^ propH ^ sH;

            //    var derZ = z ^ (hiddenOnes - z);
            //    var sZ = derZ ^ (newH - prevH) ^ sO;

            //    //The way prevH influence current state
            //    var sHnext = ((hiddenOnes - z) ^ sO);
            //    */

            //    //Matrices below (and sHnext) will be rewritten during calculation!
            //    var sH = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);
            //    var sHprop = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);
            //    var sR = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);
            //    var sZ = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);

            //    CalcBackHadamards(sH, sR, sZ, sHprop, sHnext, z, r, H, newH, prevH, sY, propH);

            //    //sHnext = (tWhh * sHprop) + (tWhr * sR) + sHnext + (tWhz * sZ);
            //    sHnext.Accumulate(_whh.Weight, sHprop, 1.0f, 1.0f, Transpose.Transpose);
            //    sHnext.Accumulate(_whr.Weight, sR, 1.0f, 1.0f, Transpose.Transpose);
            //    sHnext.Accumulate(_whz.Weight, sZ, 1.0f, 1.0f, Transpose.Transpose);

            //    if (needInputSens)
            //    {
            //        var sInput = new DenseMatrix(x.RowCount, BatchSize);
            //        sInput.Accumulate(_wxz.Weight, sZ, 1.0f, 1.0f, Transpose.Transpose);
            //        sInput.Accumulate(_wxr.Weight, sR, 1.0f, 1.0f, Transpose.Transpose);
            //        sInput.Accumulate(_wxh.Weight, sH, 1.0f, 1.0f, Transpose.Transpose);
            //        inputSensList.Insert(0, sInput);
            //    }
            //    else
            //        inputSensList.Insert(0, new DenseMatrix(x.RowCount, BatchSize));
            //    /*
            //    var dGradWhh = sHprop * tPrevH;
            //    var dGradWhr = sR * tPrevH;
            //    var dGradWhz = sZ * tPrevH;

            //    var dGradWhy = sY * Matrix.Transpose(H);

            //    var dGradWxh = sH * tX;
            //    var dGradWxr = sR * tX;
            //    var dGradWxz = sZ * tX;

            //    gradWhh += dGradWhh;
            //    gradWhr += dGradWhr;
            //    gradWhz += dGradWhz;
            //    gradWhy += dGradWhy;

            //    gradWxh += dGradWxh;
            //    gradWxr += dGradWxr;
            //    gradWxz += dGradWxz;
            //    */

            //    _whr.Gradient.Accumulate(sR, prevH, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
            //    _whz.Gradient.Accumulate(sZ, prevH, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
            //    _whh.Gradient.Accumulate(sHprop, prevH, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);

            //    _wxr.Gradient.Accumulate(sR, x, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
            //    _wxz.Gradient.Accumulate(sZ, x, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
            //    _wxh.Gradient.Accumulate(sH, x, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);

            //    if (sH.ColumnCount > 1)
            //    {
            //        _bxr.Gradient.Accumulate(sR, hiddenIdentity, 1.0f);
            //        _bxz.Gradient.Accumulate(sZ, hiddenIdentity, 1.0f);
            //        _bxh.Gradient.Accumulate(sH, hiddenIdentity, 1.0f);
            //    }
            //    else
            //    {
            //        _bxr.Gradient.Accumulate(sR);
            //        _bxz.Gradient.Accumulate(sZ);
            //        _bxh.Gradient.Accumulate(sH);
            //    }
            //}

            ////clamp gradients to this value
            //const float CLAMP = 5.0f;
            //ClampGrads(CLAMP);
            //return inputSensList;
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
        }

        public override LayerSpecBase CreateSpec()
        {
            var weights = new GruLayerWeights
                          {
                              //Wxr = _wxr.Weight,
                              //Wxz = _wxz.Weight,
                              //Wxh = _wxh.Weight,

                              //Whr = _whr.Weight,
                              //Whz = _whz.Weight,
                              //Whh = _whh.Weight,

                              //bxr = _bxr.Weight,
                              //bxz = _bxz.Weight,
                              //bxh = _bxh.Weight,

                              //bhr = _bhr.Weight,
                              //bhz = _bhz.Weight,
                              //bhh = _bhh.Weight
                          };

            return new GruLayerSpec(InputSize, BatchSize, SeqLen, 1, _hSize, weights);
        }
    }
}