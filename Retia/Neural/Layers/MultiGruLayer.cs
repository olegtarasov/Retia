using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Interop;
using Retia.Neural.Initializers;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class MultiGruLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly List<GruLayer<T>> _layers;

        public MultiGruLayer(int xSize, int hSize, int layers) : this(xSize, hSize, layers,
            new ProportionalRandomMatrixInitializer<T>(),
            new ProportionalRandomMatrixInitializer<T>(),
            new ConstantMatrixInitializer<T>())
        {
        }

        public MultiGruLayer(int xSize, int hSize, int layers,
            IMatrixInitializer<T> linearWeightInitializer,
            IMatrixInitializer<T> hiddenWeightInitializer,
            IMatrixInitializer<T> biasInitializer)
        {
            if (layers < 1) throw new ArgumentOutOfRangeException(nameof(layers), "Layer cound should be at least one!");

            _layers = new List<GruLayer<T>>();
            _layers.Add(new GruLayer<T>(xSize, hSize, linearWeightInitializer, hiddenWeightInitializer, biasInitializer));

            for (int i = 1; i < layers; i++)
            {
                _layers.Add(new GruLayer<T>(hSize, hSize, linearWeightInitializer, hiddenWeightInitializer, biasInitializer));
            }
        }

        public MultiGruLayer(MultiGruLayer<T> other) : base(other)
        {
            _layers = other._layers.ConvertAll(x => (GruLayer<T>)x.Clone());
        }

        public MultiGruLayer(BinaryReader reader) : base(reader)
        {
            _layers = new List<GruLayer<T>>();

            int count = reader.ReadInt32();
            for (int i = 0; i < count; i++)
            {
                _layers.Add(new GruLayer<T>(reader));
            }
        }

        public override int InputSize => _layers[0].InputSize;
        public override int OutputSize => _layers[0].OutputSize;
        public override int TotalParamCount => _layers.Select(x => x.TotalParamCount).Sum();
        public override IReadOnlyList<NeuroWeight<T>> Weights => _layers.SelectMany(x => x.Weights).ToList();

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> outSens, bool needInputSens = true, bool clearGrad = true)
        {
            var curSens = outSens;
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                curSens = _layers[i].BackPropagate(curSens, true, clearGrad);
            }

            return curSens;
        }

        public override void ClampGrads(float limit)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].ClampGrads(limit);
            }
        }

        public override void ClearGradients()
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].ClearGradients();
            }
        }

        public override LayerBase<T> Clone()
        {
            return new MultiGruLayer<T>(this);
        }

        public override IntPtr CreateGpuLayer()
        {
            GpuLayerPtr = GpuInterface.CreateGruLayer(InputSize, OutputSize, _layers.Count, BatchSize, SeqLen);
            TransferWeightsToDevice();

            return GpuLayerPtr;
        }

        public override List<Matrix<T>> ErrorPropagate(List<Matrix<T>> targets)
        {
            throw new NotImplementedException();
        }

        public override void FromVectorState(T[] vector, ref int idx)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].FromVectorState(vector, ref idx);
            }
        }

        public override void InitSequence()
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].InitSequence();
            }
        }

        public override void Optimize(OptimizerBase<T> optimizer)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].Optimize(optimizer);
            }
        }

        public override void ResetMemory()
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].ResetMemory();
            }
        }

        public override void ResetOptimizer()
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].ResetOptimizer();
            }
        }

        public override void Save(Stream s)
        {
            base.Save(s);

            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_layers.Count);
                for (int i = 0; i < _layers.Count; i++)
                {
                    _layers[i].Save(s);
                }
            }
        }

        public override void SetParam(int i, T value)
        {
            throw new NotImplementedException();
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            var curInput = input;
            for (int i = 0; i < _layers.Count; i++)
            {
                curInput = _layers[i].Step(curInput, inTraining);
            }

            return curInput;
        }

        public override void ToVectorState(T[] destination, ref int idx, bool grad = false)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].ToVectorState(destination, ref idx, grad);
            }
        }

        public override void TransferWeightsToDevice()
        {
            TransferWeigthsToDevice(true, Weights.ToArray());
        }

        public override void TransferWeightsToHost()
        {
            TransferWeigthsToHost(true, Weights.ToArray());
        }

        protected override void Initialize()
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].Initialize(BatchSize, SeqLen);
            }
        }
    }
}