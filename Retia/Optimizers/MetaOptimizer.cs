using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Contracts;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.Layers;

namespace Retia.Optimizers
{
    public class MetaOptimizer : OptimizerBase<float>
    {
        private readonly Dictionary<Guid, Matrix<float>[][]> _hiddenStates;
        private readonly Dictionary<Guid, List<Matrix<float>>[][]> _inputs, _outputs;
        private readonly Dictionary<Guid, List<Matrix<float>>[]> _sens;
        private readonly LayeredNet<float> _network;
        private readonly OptimizerBase<float> _optimizer;

        private const int SeqLen = 20;

        public MetaOptimizer(IReadOnlyList<NeuroWeight<float>> weights) : base(0.0f)
        {
            const int hSize = 20;

            _hiddenStates = new Dictionary<Guid, Matrix<float>[][]>();
            _inputs = new Dictionary<Guid, List<Matrix<float>>[][]>();
            _outputs = new Dictionary<Guid, List<Matrix<float>>[][]>();
            _sens = new Dictionary<Guid, List<Matrix<float>>[]>();
            foreach (var weight in weights)
            {
                var hBlock = new Matrix<float>[2][];
                var iBlock = new List<Matrix<float>>[3][];
                var oBlock = new List<Matrix<float>>[3][];
                int length = weight.Weight.Length();

                _hiddenStates[weight.Id] = hBlock;
                _inputs[weight.Id] = iBlock;
                _outputs[weight.Id] = oBlock;
                _sens[weight.Id] = Enumerable.Range(0, length).Select(x => new List<Matrix<float>>()).ToArray();
                
                for (int layerIdx = 0; layerIdx < 3; layerIdx++)
                {
                    if (layerIdx < 2)
                    {
                        hBlock[layerIdx] = Enumerable.Range(0, length).Select(x => MatrixFactory.Create<float>(hSize, 1)).ToArray();
                    }
                    iBlock[layerIdx] = Enumerable.Range(0, length).Select(x => new List<Matrix<float>>()).ToArray();
                    oBlock[layerIdx] = Enumerable.Range(0, length).Select(x => new List<Matrix<float>>()).ToArray();
                }
            }

            _optimizer = new AdamOptimizer<float>();
            _network = new LayeredNet<float>(1, SeqLen, 
                new GruLayer<float>(2, hSize), 
                new GruLayer<float>(hSize, hSize),
                new LinearLayer<float>(hSize, 1))
                       {
                           Optimizer = _optimizer
                       };
        }

        public MetaOptimizer(OptimizerBase<float> other) : base(other)
        {
        }

        public override void Optimize(NeuroWeight<float> weight)
        {
            var wa = weight.Weight.AsColumnMajorArray();
            var ga = weight.Weight.AsColumnMajorArray();
            var stateBlock = _hiddenStates[weight.Id];
            var iBlock = _inputs[weight.Id];
            var oBlock = _outputs[weight.Id];
            var sensBlock = _sens[weight.Id];

            for (int i = 0; i < wa.Length; i++)
            {
                for (int layerIdx = 0; layerIdx < 3; layerIdx++)
                {
                    var layer = _network.Layers[layerIdx];

                    if (layerIdx < 2)
                    {
                        ((GruLayer<float>)layer).HiddenState = stateBlock[layerIdx][i];
                    }

                    layer.Inputs = iBlock[layerIdx][i];
                    layer.Outputs = oBlock[layerIdx][i];
                }

                var step = _network.Step(ScaleInput(ga[i]), true);
                sensBlock[i].Add(MatrixFactory.Create<float>(1, 1, ga[i] * step[0, 0]));
                
                for (int layerIdx = 0; layerIdx < 2; layerIdx++)
                {
                    var layer = (GruLayer<float>)_network.Layers[layerIdx];
                    stateBlock[layerIdx][i] = layer.HiddenState;
                }

                ga[i] += step[0, 0] * 0.1f;
            }
        }

        public void MetaOptimize()
        {
            foreach (var layer in _network.Layers)
            {
                layer.ClearGradients();
            }

            foreach (var weightId in _hiddenStates.Keys)
            {
                var weightSens = _sens[weightId];
                var weightInputs = _inputs[weightId];
                var weightOutputs = _outputs[weightId];
                var weightHidden = _hiddenStates[weightId];

                for (int weightIdx = 0; weightIdx < weightSens.Length; weightIdx++)
                {
                    var curSens = weightSens[weightIdx];

                    for (int layerIdx = 2; layerIdx >= 0; layerIdx--)
                    {
                        var layer = _network.Layers[layerIdx];
                        if (layerIdx < 2)
                        {
                            ((GruLayer<float>)layer).HiddenState = weightHidden[layerIdx][weightIdx];
                        }

                        layer.Inputs = weightInputs[layerIdx][weightIdx];
                        layer.Outputs = weightOutputs[layerIdx][weightIdx];

                        curSens = layer.BackPropagate(curSens, true, false);

                        layer.Inputs.Clear();
                        layer.Outputs.Clear();
                    }

                    curSens.Clear();
                }
            }

            _network.Optimize();
        }

        private readonly float _threshold = (float)Math.Exp(-10.0d);
        private Matrix<float> ScaleInput(float input)
        {
            var result = MatrixFactory.Create<float>(2, 1);
            float abs = Math.Abs(input);
            if (abs > _threshold)
            { 
                result[0, 0] = ((float)Math.Log(abs) / 10.0f);
                result[1, 0] = Math.Sign(input);
            }
            else
            {
                result[0, 0] = -1.0f;
                result[1, 0] = (float)Math.Exp(10.0d) * input;
            }

            return result;
        }

        public override OptimizerSpecBase CreateSpec()
        {
            throw new NotSupportedException();
        }

        public override OptimizerBase<float> Clone()
        {
            throw new NotSupportedException();
        }
    }
}