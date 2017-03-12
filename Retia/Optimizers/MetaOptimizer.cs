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
        private readonly LayeredNet<float> _network;
        private readonly OptimizerBase<float> _optimizer;

        private const int SeqLen = 20;

        public MetaOptimizer(IReadOnlyList<NeuroWeight<float>> weights) : base(0.0f)
        {
            const int hSize = 20;

            _hiddenStates = new Dictionary<Guid, Matrix<float>[][]>();
            foreach (var weight in weights)
            {
                var block = new Matrix<float>[2][];
                _hiddenStates[weight.Id] = block;

                for (int layerIdx = 0; layerIdx < 2; layerIdx++)
                {
                    block[layerIdx] = Enumerable.Range(0, weight.Weight.Length()).Select(x => MatrixFactory.Create<float>(hSize, 1)).ToArray();
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

            var layer1 = (GruLayer<float>)_network.Layers[0];
            var layer2 = (GruLayer<float>)_network.Layers[1];

            for (int i = 0; i < wa.Length; i++)
            {
                layer1.HiddenState = stateBlock[0][i];
                layer2.HiddenState = stateBlock[1][i];

                var step = _network.Step(ScaleInput(ga[i]));

                stateBlock[0][i] = layer1.HiddenState;
                stateBlock[1][i] = layer2.HiddenState;

                ga[i] += step[0, 0] * 0.1f;
            }
        }

        public void MetaOptimize(List<Matrix<float>> sensitivities)
        {
            if (sensitivities.Count != SeqLen) throw new InvalidOperationException("Wrong sensitivity count!");

            var sens = sensitivities;
            for (int i = _network.Layers.Count - 1; i >= 0; i--)
            {
                sens = _network.Layers[i].BackPropagate(sens, true);
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