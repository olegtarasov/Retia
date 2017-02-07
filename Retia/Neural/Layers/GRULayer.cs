using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Providers.LinearAlgebra;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class GruLayer : NeuroLayer
    {
        private const double Dispersion = 5e-2;

        private readonly List<Matrix> _hNewVals = new List<Matrix>();
        private readonly List<Matrix> _hPropVals = new List<Matrix>();

        private readonly List<Matrix> _rVals = new List<Matrix>();
        private readonly List<Matrix> _zVals = new List<Matrix>();

        private readonly int _hSize;

        private NeuroWeight _whh, _whr, _whz;
        private NeuroWeight _wxh, _wxr, _wxz;
        private NeuroWeight _bxh, _bxr, _bxz;
        private NeuroWeight _bhh, _bhr, _bhz;
        private Matrix _hiddenOnes;

        /// <summary>
        ///     Cache of last calculated hidden output.
        ///     Initially zero matrix
        /// </summary>
        private Matrix _lastH;

        public override Matrix[] InternalState
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
            new ProportionalRandomMatrixInitializer(),
            new ProportionalRandomMatrixInitializer(),
            new ConstantMatrixInitializer {Value = 0.0f})
        {
        }

        public GruLayer(int xSize, int hSize, 
            IMatrixInitializer linearWeightInitializer,
            IMatrixInitializer hiddenWeightInitializer,
            IMatrixInitializer biasInitializer)
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

            Initialize(1, 1);
            
            ResetOptimizer();
        }

        public GruLayer(BinaryReader reader)
        {
            _bxr = NeuroWeight.Load(reader.BaseStream);
            _bxz = NeuroWeight.Load(reader.BaseStream);
            _bxh = NeuroWeight.Load(reader.BaseStream);

            _bhr = NeuroWeight.Load(reader.BaseStream);
            _bhz = NeuroWeight.Load(reader.BaseStream);
            _bhh = NeuroWeight.Load(reader.BaseStream);

            _wxr = NeuroWeight.Load(reader.BaseStream);
            _wxz = NeuroWeight.Load(reader.BaseStream);
            _wxh = NeuroWeight.Load(reader.BaseStream);

            _whr = NeuroWeight.Load(reader.BaseStream);
            _whz = NeuroWeight.Load(reader.BaseStream);
            _whh = NeuroWeight.Load(reader.BaseStream);

            _lastH = MatrixFactory.Load(reader.BaseStream);
            _hiddenOnes = DenseMatrix.Create(_hSize, _lastH.ColumnCount, 1.0f);
        }

        private GruLayer(GruLayer other) : base(other)
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
        }

        public override int InputSize => _wxh.Weight.ColumnCount;
        public override int OutputSize => _whh.Weight.RowCount;

        // TODO: Clean this shit up
        public override int TotalParamCount => _wxr.Weight.AsColumnMajorArray().Length + _wxz.Weight.AsColumnMajorArray().Length + _wxh.Weight.AsColumnMajorArray().Length +
                                               _whr.Weight.AsColumnMajorArray().Length + _whz.Weight.AsColumnMajorArray().Length + _whh.Weight.AsColumnMajorArray().Length +
                                               _bxr.Weight.AsColumnMajorArray().Length + _bxz.Weight.AsColumnMajorArray().Length + _bxh.Weight.AsColumnMajorArray().Length;

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


        public override NeuroLayer Clone()
        {
            return new GruLayer(this);
        }

        public override void Optimize(OptimizerBase optimizer)
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
            _lastH = new DenseMatrix(_lastH.RowCount, _lastH.ColumnCount);
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

        private unsafe void CalculateHSlow(Matrix H, Matrix hCandidate, Matrix z, Matrix lastH)
        {
            var ha = H.AsColumnMajorArray();
            var hcana = hCandidate.AsColumnMajorArray();
            var za = z.AsColumnMajorArray();
            var lastha = lastH.AsColumnMajorArray();

            fixed (float* pArr = ha, pCanArr = hcana, pzArr = za, plastHArr = lastha)
            {
                ParallelFor.Instance.Execute(CalculateHSlow, ha.Length, new void*[]{pArr, pCanArr, pzArr, plastHArr});
            }
        }

        private unsafe void CalculateHSlow(int startIdx, int endIdx, void*[] ptrs)
        {
            float* pArr = (float*)ptrs[0], pCanArr = (float*)ptrs[1], pzArr = (float*)ptrs[2], plastHArr = (float*)ptrs[3];

            for (int i = startIdx; i < endIdx; i++)
            {
                float* arrPtr = pArr + i, hCanArrPtr = pCanArr + i, zArrPtr = pzArr + i, lastHPtr = plastHArr + i;
                float zEl = *zArrPtr;
                *arrPtr = (1 - zEl) * *hCanArrPtr + zEl * *lastHPtr;
            }
        }

        protected override void Initialize()
        {
            _lastH = new DenseMatrix(_hSize, BatchSize);
            _hiddenOnes = DenseMatrix.Create(_hSize, BatchSize, DenseMatrix.One);
            InitBackPropagation();
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            if (input.RowCount != _wxh.Weight.ColumnCount)
                throw new Exception($"Wrong input matrix row size provided!\nExpected: {_wxh.Weight.ColumnCount}, got: {input.RowCount}");
            if (input.ColumnCount != BatchSize)
                throw new Exception($"Wrong input batch size!\nExpected: {BatchSize}, got: {input.ColumnCount}");

            //var z = Bz + Wxz*input + Whz*lastH;
            var z = (Matrix)(_bxz.Weight.TileColumns(BatchSize) + _bhz.Weight.TileColumns(BatchSize));
            z.Accumulate(_wxz.Weight, input, 1.0f);
            z.Accumulate(_whz.Weight, _lastH, 1.0f);


            //var r = Br + Wxr*input + Whr*lastH;
            var r = (Matrix)(_bxr.Weight.TileColumns(BatchSize) + _bhr.Weight.TileColumns(BatchSize));
            r.Accumulate(_wxr.Weight, input, 1.0f);
            r.Accumulate(_whr.Weight, _lastH, 1.0f);

            //Sigmoid(z);
            //Sigmoid(r);
            //ActivationFuncs.ApplySigmoid(r);
            //ActivationFuncs.ApplySigmoid(z);
            ActivationFuncs.ApplySigmoid2(r, z);
            //ApplySigmoid(r, z);


            var hNew = _bxh.Weight.TileColumns(BatchSize);
            hNew.Accumulate(_wxh.Weight, input, 1.0f);

            var hProp = _bhh.Weight.TileColumns(BatchSize);
            hProp.Accumulate(_whh.Weight, _lastH);
            
            hNew = (Matrix)(hNew + (Matrix)r.PointwiseMultiply(hProp));
            ActivationFuncs.ApplyTanh(hNew);
            //ApplyTanh(hNew);

            //var H = (z ^ hNew) + ((_hiddenOnes - z) ^ _lastH);
            var H = new DenseMatrix(hNew.RowCount, hNew.ColumnCount);
            CalculateHSlow(H, hNew, z, _lastH);

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

        private void CheckInitialized()
        {
            if (_lastH == null || _hiddenOnes == null
                || _lastH.RowCount != _hSize || _hiddenOnes.RowCount != _hSize
                || _lastH.ColumnCount != BatchSize || _hiddenOnes.ColumnCount != BatchSize)
            {
                throw new InvalidOperationException("GRU layer is not initialized correctly. Call Initialize().");
            }
        }

        private void CalcBackHadamards(Matrix sH, Matrix sR, Matrix sZ, Matrix sHprop, Matrix sHnext, Matrix z,
                                       Matrix r, Matrix h,
                                       Matrix newH,
                                       Matrix prevH, Matrix sY, Matrix propH)
        {
            var _sH = sH.AsColumnMajorArray();
            var _sR = sR.AsColumnMajorArray();
            var _sZ = sZ.AsColumnMajorArray();
            var _sHnext = sHnext.AsColumnMajorArray();
            var _sHprop = sHprop.AsColumnMajorArray();
            var _z = z.AsColumnMajorArray();
            var _r = r.AsColumnMajorArray();
            var _h = h.AsColumnMajorArray();
            var _prevH = prevH.AsColumnMajorArray();
            var _sY = sY.AsColumnMajorArray();
            var _propH = propH.AsColumnMajorArray();
            var _newH = newH.AsColumnMajorArray();

            /*
            var sO = sHnext + sY;
            var derH = hiddenOnes - (newH ^ newH);
            var sH = derH ^ z ^ sO;
            var sHprop = sH ^ r;

            var derR = r ^ (hiddenOnes - r);
            var sR = derR ^ propH ^ sH;

            var derZ = z ^ (hiddenOnes - z);
            var sZ = derZ ^ (newH - prevH) ^ sO;

            //The way prevH influence current state
            sHnext = ((hiddenOnes - z) ^ sO);
            */

            for (var i = 0; i < _hiddenOnes.RowCount * _hiddenOnes.ColumnCount; i++)
            {
                var sO = _sHnext[i] + _sY[i];
                var derH = 1 - _newH[i] * _newH[i];
                _sH[i] = derH * _z[i] * sO;
                _sHprop[i] = _sH[i] * _r[i];
                var derR = _r[i] * (1.0f - _r[i]);
                _sR[i] = derR * _propH[i] * _sH[i];
                var derZ = _z[i] * (1.0f - _z[i]);
                _sZ[i] = derZ * (_newH[i] - _prevH[i]) * sO;
                _sHnext[i] = (1 - _z[i]) * sO;
            }
        }

        public override List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true)
        {
            if (Outputs.Count != Inputs.Count)
                throw new Exception("Backprop was not initialized (empty state sequence)");
            if (Inputs.Count == 0)
                throw new Exception("Empty inputs history, nothing to propagate!");
            if (outSens.Count != Inputs.Count)
                throw new Exception("Not enough sensitivies in list!");

            var inputSensList = new List<Matrix>(SeqLen);

            var hiddenIdentity = DenseMatrix.Create(BatchSize, 1, DenseMatrix.One);
            var yIdentity = DenseMatrix.Create(BatchSize, 1, DenseMatrix.One);


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


            //Sensitivity of next state
            var sHnext = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);

            for (var i = Inputs.Count - 1; i >= 0; i--)
            {
                //output sensitivity
                var sY = outSens[i];

                //inputs
                var x = Inputs[i];

                //R matrix
                var r = _rVals[i];

                //Z matrix
                var z = _zVals[i];

                //Previous hidden value

                var prevH = i > 0 ? Outputs[i - 1] : new DenseMatrix(OutputSize, BatchSize);

                //Current hidden value
                var H = Outputs[i];

                //Weighted previous hidden value
                var propH = _hPropVals[i];

                //Current hidden candidate
                var newH = _hNewVals[i];


                //Transponsed martices
                /*
                var tPrevH = Matrix.Transpose(prevH);
                var tX = Matrix.Transpose(x);
                */

                //Sigmoid derrivative:  f'(x) = f(x)*(1-f(x))
                //Tanh derrivative:     f'(x) = 1-f(x)*f(x)

                //var sO = sHnext + sY;


                /*
                var derH = hiddenOnes - (newH ^ newH);
                var sH = derH ^ z ^ sO;

                var sHprop = sH ^ r;

                var derR = r ^ (hiddenOnes - r);
                var sR = derR ^ propH ^ sH;

                var derZ = z ^ (hiddenOnes - z);
                var sZ = derZ ^ (newH - prevH) ^ sO;

                //The way prevH influence current state
                var sHnext = ((hiddenOnes - z) ^ sO);
                */

                //Matrices below (and sHnext) will be rewritten during calculation!
                var sH = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);
                var sHprop = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);
                var sR = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);
                var sZ = new DenseMatrix(_hiddenOnes.RowCount, _hiddenOnes.ColumnCount);

                CalcBackHadamards(sH, sR, sZ, sHprop, sHnext, z, r, H, newH, prevH, sY, propH);

                //sHnext = (tWhh * sHprop) + (tWhr * sR) + sHnext + (tWhz * sZ);
                sHnext.Accumulate(_whh.Weight, sHprop, 1.0f, 1.0f, Transpose.Transpose);
                sHnext.Accumulate(_whr.Weight, sR, 1.0f, 1.0f, Transpose.Transpose);
                sHnext.Accumulate(_whz.Weight, sZ, 1.0f, 1.0f, Transpose.Transpose);

                if (needInputSens)
                {
                    var sInput = new DenseMatrix(x.RowCount, BatchSize);
                    sInput.Accumulate(_wxz.Weight, sZ, 1.0f, 1.0f, Transpose.Transpose);
                    sInput.Accumulate(_wxr.Weight, sR, 1.0f, 1.0f, Transpose.Transpose);
                    sInput.Accumulate(_wxh.Weight, sH, 1.0f, 1.0f, Transpose.Transpose);
                    inputSensList.Insert(0, sInput);
                }
                else
                    inputSensList.Insert(0, new DenseMatrix(x.RowCount, BatchSize));
                /*
                var dGradWhh = sHprop * tPrevH;
                var dGradWhr = sR * tPrevH;
                var dGradWhz = sZ * tPrevH;

                var dGradWhy = sY * Matrix.Transpose(H);

                var dGradWxh = sH * tX;
                var dGradWxr = sR * tX;
                var dGradWxz = sZ * tX;

                gradWhh += dGradWhh;
                gradWhr += dGradWhr;
                gradWhz += dGradWhz;
                gradWhy += dGradWhy;

                gradWxh += dGradWxh;
                gradWxr += dGradWxr;
                gradWxz += dGradWxz;
                */

                _whr.Gradient.Accumulate(sR, prevH, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
                _whz.Gradient.Accumulate(sZ, prevH, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
                _whh.Gradient.Accumulate(sHprop, prevH, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);

                _wxr.Gradient.Accumulate(sR, x, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
                _wxz.Gradient.Accumulate(sZ, x, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);
                _wxh.Gradient.Accumulate(sH, x, 1.0f, 1.0f, Transpose.DontTranspose, Transpose.Transpose);

                if (sH.ColumnCount > 1)
                {
                    _bxr.Gradient.Accumulate(sR, hiddenIdentity, 1.0f);
                    _bxz.Gradient.Accumulate(sZ, hiddenIdentity, 1.0f);
                    _bxh.Gradient.Accumulate(sH, hiddenIdentity, 1.0f);
                }
                else
                {
                    _bxr.Gradient.Accumulate(sR);
                    _bxz.Gradient.Accumulate(sZ);
                    _bxh.Gradient.Accumulate(sH);
                }
            }

            //clamp gradients to this value
            const float CLAMP = 5.0f;
            ClampGrads(CLAMP);
            return inputSensList;
        }

        public override void ClampGrads(float limit)
        {

            _whr.Gradient.Clamp(-limit, limit);
            _whz.Gradient.Clamp(-limit, limit);
            _whh.Gradient.Clamp(-limit, limit);


            _wxr.Gradient.Clamp(-limit, limit);
            _wxz.Gradient.Clamp(-limit, limit);
            _wxh.Gradient.Clamp(-limit, limit);


            _bxr.Gradient.Clamp(-limit, limit);
            _bxz.Gradient.Clamp(-limit, limit);
            _bxh.Gradient.Clamp(-limit, limit);

            _bhr.Gradient.Clamp(-limit, limit);
            _bhz.Gradient.Clamp(-limit, limit);
            _bhh.Gradient.Clamp(-limit, limit);
        }

        public override void ToVectorState(float[] destination, ref int idx, bool grad = false)
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

        public override void FromVectorState(float[] vector, ref int idx)
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


        public override void InitBackPropagation()
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
                              Wxr = _wxr.Weight,
                              Wxz = _wxz.Weight,
                              Wxh = _wxh.Weight,

                              Whr = _whr.Weight,
                              Whz = _whz.Weight,
                              Whh = _whh.Weight,

                              bxr = _bxr.Weight,
                              bxz = _bxz.Weight,
                              bxh = _bxh.Weight,

                              bhr = _bhr.Weight,
                              bhz = _bhz.Weight,
                              bhh = _bhh.Weight
                          };

            return new GruLayerSpec(InputSize, BatchSize, SeqLen, 1, _hSize, weights);
        }
    }
}