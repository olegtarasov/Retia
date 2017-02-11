using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Mathematics;
using Retia.NativeWrapper;
using Retia.Neural.Layers;
using Retia.Optimizers;

namespace Retia.Neural
{
    public class LayeredNet : NeuralNet, IDisposable
    {
        private const byte LayerMagic = 0xBA;
        //public static uint cnt = 0;
        //public readonly uint id; 
        private static readonly byte[] _magic = {0xDE, 0xAD, 0xCA, 0xFE};

        protected readonly int BatchSize, SeqLen;
        protected readonly List<NeuroLayer> Layers = new List<NeuroLayer>();

        private GpuNetwork _gpuNetwork;

        public LayeredNet(int batchSize, int seqLen, params NeuroLayer[] layers)
        {
            if (layers.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(layers));

            BatchSize = batchSize;
            SeqLen = seqLen;

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];
                Layers.Add(layer);
                layer.Initialize(batchSize, seqLen);
                if (i == 0)
                    continue;
               if (layers[i-1].OutputSize != layer.InputSize)
                    throw new ArgumentException($"Dimension of layer #{i} and #{i+1} does not agree ({layers[i-1].OutputSize}!={layer.InputSize})!");
            }
            
        }

        protected LayeredNet(LayeredNet other)
        {
            BatchSize = other.BatchSize;
            SeqLen = other.SeqLen;
            Layers = other.Layers.Select(x => x.Clone()).ToList();
        }

        private LayeredNet()
        {
        }

        public override int InputSize => InLayer.InputSize;
        public override int OutputSize => OutLayer.OutputSize;

        public override int TotalParamCount
        {
            get
            {
                var cnt = 0;
                foreach (var layer in Layers)
                    cnt += layer.TotalParamCount;
                return cnt;
            }
        }


        protected NeuroLayer OutLayer => Layers[Layers.Count - 1];
        protected NeuroLayer InLayer => Layers[0];

        public static LayeredNet Load(string path)
        {
            return StreamHelpers.LoadObject(path, Load);
        }

        public static LayeredNet Load(Stream stream)
        {
            return Load<LayeredNet>(stream);
        }

        public static T Load<T>(string path) where T : LayeredNet
        {
            return StreamHelpers.LoadObject(path, Load<T>);
        }

        public static T Load<T>(Stream stream) where T : LayeredNet
        {
            using (var reader = stream.NonGreedyReader())
            {
                var magic = reader.ReadBytes(_magic.Length);
                if (!magic.SequenceEqual(_magic))
                {
                    throw new InvalidOperationException("Invalid magic bytes!");
                }

                int layerCount = reader.ReadInt32();
                var layers = new NeuroLayer[layerCount];
                for (int i = 0; i < layerCount; i++)
                {
                    if (reader.ReadByte() != LayerMagic)
                    {
                        throw new InvalidOperationException("Invalid layer magic!");
                    }

                    string typeName = reader.ReadString();
                    if (string.IsNullOrEmpty(typeName))
                    {
                        throw new InvalidOperationException("Invalid type name!");
                    }

                    var layerType = Type.GetType(typeName);
                    if (layerType == null)
                    {
                        throw new InvalidOperationException($"Can't find layer type {typeName}");
                    }

                    var layer = (NeuroLayer)Activator.CreateInstance(layerType, reader);
                    layers[i] = layer;
                }

                return (T)Activator.CreateInstance(typeof(T), new object[] {layers});
            }
        }

        public static void CheckGrad(LayeredNet net, int seqLen)
        {
            Console.WriteLine("Starting grad check");
            float delta = 1e-3f;

            net.ResetMemory();

            var inputs = new List<Matrix>(seqLen);
            var targets = new List<Matrix>(seqLen);
            for (int i = 0; i < seqLen; i++)
            {
                var randomInput = (Matrix)DenseMatrix.CreateRandom(net.InputSize, 1, new Normal(0.0f, 2.0f));
                var randomTarget = (Matrix)DenseMatrix.CreateRandom(net.OutputSize, 1, new Normal(0.0f, 2.0f)); 
                randomTarget = SoftMax.SoftMaxNorm(randomTarget);
                inputs.Add(randomInput);
                targets.Add(randomTarget);
            }

            var controlNet = new LayeredNet(net);
            controlNet.TrainSequence(inputs, targets);
            var hasErr = false;
            for (int i = 0; i < net.TotalParamCount; i++)
            {
                var netP = new LayeredNet(net);
                var netN = new LayeredNet(net);
                netP.SetParam(i, netP.GetParam(i) + delta);
                netN.SetParam(i, netN.GetParam(i) - delta);

                double errP=0.0, errN=0.0;
                for (int s = 0; s < seqLen; s++)
                {
                    var pY=netP.Step(inputs[s]);
                    errP += netP.Error(pY, targets[s]);

                    var nY = netN.Step(inputs[s]);
                    errN += netN.Error(nY, targets[s]);
                }
                var numGrad = (errP - errN)/(2*delta);
                var grad = controlNet.GetParam(i, true);
                var d = grad - numGrad;
                if (Math.Abs(d) > 1e-7)
                {
                    Console.WriteLine($"Grad err={d} in param {i}");
                    hasErr = true;
                }
            }
            Console.WriteLine(hasErr ? "Grad check complete with ERRORS!" : "Grad check OK!");
        }

        public override double TrainSequence(List<Matrix> inputs, List<Matrix> targets)
        {
            if (_gpuNetwork != null)
            {
                //return _gpuNetwork.TrainSequence(inputs, targets);
            }

            return base.TrainSequence(inputs, targets);
        }

        public void UseGpu()
        {
            if (_gpuNetwork != null)
            {
                return;
            }

            _gpuNetwork = new GpuNetwork(CreateSpec());
        }

        public void UseCpu()
        {
            _gpuNetwork?.Dispose();
            _gpuNetwork = null;
        }

        public override void Save(Stream s)
        {
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_magic);
                writer.Write(Layers.Count);

                foreach (var layer in Layers)
                {
                    writer.Write(LayerMagic);
                    writer.Write(layer.GetType().AssemblyQualifiedName);
                    layer.Save(s);
                }
            }
        }

        public override NeuralNet Clone()
        {
            return new LayeredNet(this);
        }

        public override void Optimize()
        {
            if (_gpuNetwork != null)
            {
                _gpuNetwork.Optimize();
                return;
            }

            foreach (var layer in Layers)
                layer.Optimize(Optimizer);
        }

        public override double Error(Matrix y, Matrix target)
        {
            return OutLayer.LayerError(y, target);
        }

        public override List<Matrix> BackPropagate(List<Matrix> targets, bool needInputSens = false)
        {  
            List<Matrix> prop = OutLayer.ErrorPropagate(targets);
            if (Layers.Count < 2)
                return prop;
            for (int i = Layers.Count - 2; i > 0; i--)
            {
                var layer = Layers[i];
                prop = layer.BackPropagate(prop, true);
            }
            return InLayer.BackPropagate(prop, needInputSens);
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            var prop = input;
            foreach (var layer in Layers)
                prop = layer.Step(prop, inTraining);
            return prop;
        }

        public override void ResetMemory()
        {
            if (_gpuNetwork != null)
            {
                _gpuNetwork.ResetMemory();
                return;
            }

            foreach (var layer in Layers)
                layer.ResetMemory();
        }

        public override void ResetOptimizer()
        {
            if (_gpuNetwork != null)
            {
                _gpuNetwork.ResetOptimizerCache();
                return;
            }

            foreach (var layer in Layers)
                layer.ResetOptimizer();
        }

        public override void InitSequence()
        {
            foreach (var layer in Layers)
                layer.InitSequence();
        }

        private void SetParam(int i, double value)
        {
            if (i >= TotalParamCount)
                throw new ArgumentException($"Parameter index ({i}) should be less than {TotalParamCount}");

            var paramCnt = 0;
            foreach (var layer in Layers)
            {
                if (i < paramCnt + layer.TotalParamCount)
                {
                    layer.SetParam(i - paramCnt, value);
                    return;
                }
                paramCnt += layer.TotalParamCount;
            }
            throw new Exception($"What the fuck is this? Your index={i} is somehow less then TotalParamCount={TotalParamCount} but more than sum of all layer param counts {paramCnt}!");
        }

        private double GetParam(int i, bool grad=false)
        {
            if(i>=TotalParamCount)
                throw new ArgumentException($"Parameter index ({i}) should be less than {TotalParamCount}");

            var paramCnt = 0;
            foreach (var layer in Layers)
            {
                if (i < paramCnt + layer.TotalParamCount)
                    return layer.GetParam(i - paramCnt, grad);
                paramCnt += layer.TotalParamCount;
            }
            throw new Exception($"What the fuck is this? Your index={i} is somehow less then TotalParamCount={TotalParamCount} but more than sum of all layer param counts {paramCnt}!");
        }

        private LayeredNetSpec CreateSpec()
        {
            if (Optimizer == null) throw new InvalidOperationException("Set optimizer first!");
            if (Layers.Count == 0) throw new InvalidOperationException("Add some layers!");

            var result = new LayeredNetSpec(Optimizer.CreateSpec(), Layers[0].InputSize, Layers[Layers.Count - 1].OutputSize, BatchSize, SeqLen);
            LayerSpecBase lastSpec = null;
            
            foreach (var layer in Layers)
            {
                var lastGru = lastSpec as GruLayerSpec;
                if (layer is GruLayer && lastGru != null)
                {
                    var spec = (GruLayerSpec)layer.CreateSpec();
                    if (lastGru.HSize == spec.HSize)
                    {
                        lastGru.Layers++;
                        lastGru.Weights.Add(spec.Weights[0]);
                        continue;
                    }
                }

                if (lastSpec != null)
                {
                    result.Layers.Add(lastSpec);
                }

                lastSpec = layer.CreateSpec();
            }

            result.Layers.Add(lastSpec);

            return result;
        }

        #region Candidates for removal

        public override List<Matrix[]> InternalState
        {
            get
            {
                var result = new List<Matrix[]>(Layers.Count);
                foreach (var layer in Layers)
                {
                    result.Add(layer.InternalState);
                }
                return result;
            }
            set
            {
                if (value.Count() != Layers.Count)
                    throw new Exception($"Internal state of {GetType().AssemblyQualifiedName} should consist of {Layers.Count} Matrix[]");
                for (int i = 0; i < Layers.Count; i++)
                {
                    var layer = Layers[i];
                    var state = value[i];
                    layer.InternalState = state;
                }
            }
        }

        public void Dispose()
        {
            _gpuNetwork?.Dispose();
        }

        #endregion
    }
}