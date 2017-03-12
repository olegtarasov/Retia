using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Mathematics;
#if !CPUONLY
using Retia.NativeWrapper;
#endif
using Retia.Neural.Layers;
using Retia.Optimizers;

namespace Retia.Neural
{
    public class LayeredNet<T> : NeuralNet<T>, IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private const byte LayerMagic = 0xBA;
        //public static uint cnt = 0;
        //public readonly uint id; 
        private static readonly byte[] _magic = {0xDE, 0xAD, 0xCA, 0xFE};

        protected readonly int BatchSize, SeqLen;
        protected readonly List<LayerBase<T>> Layers = new List<LayerBase<T>>();

#if !CPUONLY
        private GpuNetwork _gpuNetwork;
#endif
        private OptimizerBase<T> _optimizer;

        public LayeredNet(int batchSize, int seqLen, params LayerBase<T>[] layers)
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

        protected LayeredNet(LayeredNet<T> other)
        {
            BatchSize = other.BatchSize;
            SeqLen = other.SeqLen;
            Layers = other.Layers.Select(x => x.Clone()).ToList();
        }

        protected LayeredNet(LayeredNet<T> other, int batchSize, int seqLength)
        {
            BatchSize = batchSize;
            SeqLen = seqLength;
            Layers = other.Layers.Select(x => x.Clone()).ToList();

            foreach (var layer in Layers)
            {
                layer.Initialize(batchSize, seqLength);
            }
        }

        private LayeredNet()
        {
        }

        public override int InputSize => InLayer.InputSize;
        public override int OutputSize => OutLayer.OutputSize;

        public override OptimizerBase<T> Optimizer
        {
            get { return _optimizer; }
            set
            {
                _optimizer = value;
#if !CPUONLY
                _optimizer.GpuOptimizer = _gpuNetwork;
#endif
            }
        }

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


        protected LayerBase<T> OutLayer => Layers[Layers.Count - 1];
        protected LayerBase<T> InLayer => Layers[0];

        public static LayeredNet<T> Load(string path)
        {
            return StreamHelpers.LoadObject(path, Load);
        }

        public static LayeredNet<T> Load(Stream stream)
        {
            return Load<LayeredNet<T>>(stream);
        }

        public static TNet Load<TNet>(string path) where TNet : LayeredNet<T>
        {
            return StreamHelpers.LoadObject(path, Load<TNet>);
        }

        public static TNet Load<TNet>(Stream stream) where TNet : LayeredNet<T>
        {
            using (var reader = stream.NonGreedyReader())
            {
                var magic = reader.ReadBytes(_magic.Length);
                if (!magic.SequenceEqual(_magic))
                {
                    throw new InvalidOperationException("Invalid magic bytes!");
                }

                int batchSize = reader.ReadInt32();
                int seqLen = reader.ReadInt32();
                int layerCount = reader.ReadInt32();
                var layers = new LayerBase<T>[layerCount];
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

                    var layer = (LayerBase<T>)Activator.CreateInstance(layerType, reader);
                    layers[i] = layer;
                }

                return (TNet)Activator.CreateInstance(typeof(TNet), batchSize, seqLen, layers);
            }
        }

        //public static void CheckGrad(LayeredNet<T> net, int seqLen)
        //{
        //    Console.WriteLine("Starting grad check");
        //    float delta = 1e-3f;

        //    net.ResetMemory();

        //    var inputs = new List<Matrix<T>>(seqLen);
        //    var targets = new List<Matrix<T>>(seqLen);
        //    for (int i = 0; i < seqLen; i++)
        //    {
        //        var randomInput = MatrixFactory.RandomMatrix<T>(net.InputSize, 1, 2.0f);
        //        var randomTarget = MatrixFactory.RandomMatrix<T>(net.OutputSize, 1, 2.0f); 
        //        randomTarget = MathProvider<T>.Instance.SoftMaxNorm(randomTarget);
        //        inputs.Add(randomInput);
        //        targets.Add(randomTarget);
        //    }

        //    var controlNet = new LayeredNet<T>(net);
        //    controlNet.TrainSequence(inputs, targets);
        //    var hasErr = false;
        //    for (int i = 0; i < net.TotalParamCount; i++)
        //    {
        //        var netP = new LayeredNet<T>(net);
        //        var netN = new LayeredNet<T>(net);
        //        netP.SetParam(i, netP.GetParam(i) + delta);
        //        netN.SetParam(i, netN.GetParam(i) - delta);

        //        double errP=0.0, errN=0.0;
        //        for (int s = 0; s < seqLen; s++)
        //        {
        //            var pY=netP.Step(inputs[s]);
        //            errP += netP.Error(pY, targets[s]);

        //            var nY = netN.Step(inputs[s]);
        //            errN += netN.Error(nY, targets[s]);
        //        }
        //        var numGrad = (errP - errN)/(2*delta);
        //        var grad = controlNet.GetParam(i, true);
        //        var d = grad - numGrad;
        //        if (Math.Abs(d) > 1e-7)
        //        {
        //            Console.WriteLine($"Grad err={d} in param {i}");
        //            hasErr = true;
        //        }
        //    }
        //    Console.WriteLine(hasErr ? "Grad check complete with ERRORS!" : "Grad check OK!");
        //}

        public override double TrainSequence(List<Matrix<T>> inputs, List<Matrix<T>> targets)
        {
#if !CPUONLY
            if (_gpuNetwork != null)
            {
                if (typeof(T) != typeof(float))
                {
                    throw new InvalidOperationException("GPU is only supported for float data type!");
                }

                return _gpuNetwork.TrainSequence(inputs.Cast<Matrix<float>>().ToList(), targets.Cast<Matrix<float>>().ToList());
            }
#endif

            return base.TrainSequence(inputs, targets);
        }

        public void TransferStateToHost()
        {
#if !CPUONLY
            if (_gpuNetwork == null)
            {
                throw new InvalidOperationException("You are not using GPU!");
            }

            var spec = CreateSpec();
            _gpuNetwork.TransferStatesToHost(spec);
#else
            throw new InvalidOperationException("Library was compiled without GPU support!");
#endif
        }

        public void UseGpu()
        {
#if !CPUONLY
            if (_gpuNetwork != null)
            {
                return;
            }

            _gpuNetwork = new GpuNetwork(CreateSpec());
            _optimizer.GpuOptimizer = _gpuNetwork;
#else
            throw new InvalidOperationException("Library was compiled without GPU support!");
#endif
        }

        public void UseCpu()
        {
#if !CPUONLY
            _gpuNetwork?.Dispose();
            _gpuNetwork = null;
            _optimizer.GpuOptimizer = null;
#endif
        }

        public override void Save(Stream s)
        {
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_magic);
                writer.Write(BatchSize);
                writer.Write(SeqLen);
                writer.Write(Layers.Count);

                foreach (var layer in Layers)
                {
                    writer.Write(LayerMagic);
                    writer.Write(layer.GetType().AssemblyQualifiedName);
                    layer.Save(s);
                }
            }
        }

        public override NeuralNet<T> Clone()
        {
            return new LayeredNet<T>(this);
        }

        /// <summary>
        /// Clones current network and initializes the new network with specified batch size and sequence length.
        /// </summary>
        /// <param name="batchSize">New batch size.</param>
        /// <param name="seqLength">New sequence length.</param>
        /// <returns>New layered network which is totally decoupled from the source network.</returns>
        public LayeredNet<T> Clone(int batchSize, int seqLength)
        {
            return new LayeredNet<T>(this, batchSize, seqLength);
        }

        public override void Optimize()
        {
#if !CPUONLY
            if (_gpuNetwork != null)
            {
                _gpuNetwork.Optimize();
                return;
            }
#endif

            foreach (var layer in Layers)
                layer.Optimize(Optimizer);
        }

        public override double Error(Matrix<T> y, Matrix<T> target)
        {
            return OutLayer.LayerError(y, target);
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> targets, bool needInputSens = false)
        {  
            List<Matrix<T>> prop = OutLayer.ErrorPropagate(targets);
            if (Layers.Count < 2)
                return prop;
            for (int i = Layers.Count - 2; i > 0; i--)
            {
                var layer = Layers[i];
                prop = layer.BackPropagate(prop, true);
            }
            return InLayer.BackPropagate(prop, needInputSens);
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            var prop = input;
            foreach (var layer in Layers)
                prop = layer.Step(prop, inTraining);
            return prop;
        }

        public override void ResetMemory()
        {
#if !CPUONLY
            if (_gpuNetwork != null)
            {
                _gpuNetwork.ResetMemory();
                return;
            }
#endif

            foreach (var layer in Layers)
                layer.ResetMemory();
        }

        public override void ResetOptimizer()
        {
#if !CPUONLY
            if (_gpuNetwork != null)
            {
                _gpuNetwork.ResetOptimizerCache();
                return;
            }
#endif

            foreach (var layer in Layers)
                layer.ResetOptimizer();
        }

        public override void InitSequence()
        {
            foreach (var layer in Layers)
                layer.InitSequence();
        }

        private void SetParam(int i, T value)
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

        private T GetParam(int i, bool grad=false)
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
                if (layer is GruLayer<T> && lastGru != null)
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

        public void Dispose()
        {
#if !CPUONLY
            _gpuNetwork?.Dispose();
#endif
        }

#region Candidates for removal

        public override List<Matrix<T>[]> InternalState
        {
            get
            {
                var result = new List<Matrix<T>[]>(Layers.Count);
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

#endregion
    }
}