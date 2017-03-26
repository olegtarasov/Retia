using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Interop;
using Retia.Helpers;
using Retia.Neural.Layers;
using Retia.Optimizers;
#if !CPUONLY
#endif

namespace Retia.Neural
{
    public class LayeredNet<T> : NeuralNet<T>, IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private const byte LayerMagic = 0xBA;
        //public static uint cnt = 0;
        //public readonly uint id; 
        private static readonly byte[] _magic = {0xDE, 0xAD, 0xCA, 0xFE};

        protected readonly int BatchSize, SeqLen;
        protected readonly List<LayerBase<T>> LayersList = new List<LayerBase<T>>();

        private IntPtr _gpuNetworkPtr = IntPtr.Zero;
        private OptimizerBase<T> _optimizer;

        public LayeredNet(int batchSize, int seqLen, params LayerBase<T>[] layers)
        {
            if (layers.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(layers));

            BatchSize = batchSize;
            SeqLen = seqLen;

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];
                LayersList.Add(layer);
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
            LayersList = other.LayersList.Select(x => x.Clone()).ToList();
        }

        protected LayeredNet(LayeredNet<T> other, int batchSize, int seqLength)
        {
            BatchSize = batchSize;
            SeqLen = seqLength;
            LayersList = other.LayersList.Select(x => x.Clone()).ToList();

            foreach (var layer in LayersList)
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
                
#endif
            }
        }

        public override int TotalParamCount
        {
            get
            {
                var cnt = 0;
                foreach (var layer in LayersList)
                    cnt += layer.TotalParamCount;
                return cnt;
            }
        }

        public IReadOnlyList<LayerBase<T>> Layers => LayersList;
 
        protected LayerBase<T> OutLayer => LayersList[LayersList.Count - 1];
        protected LayerBase<T> InLayer => LayersList[0];

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

        public override unsafe double TrainSequence(List<Matrix<T>> inputs, List<Matrix<T>> targets)
        {
#if !CPUONLY
            if (_gpuNetworkPtr != IntPtr.Zero)
            {
                if (typeof(T) != typeof(float))
                {
                    throw new InvalidOperationException("GPU is only supported for float data type!");
                }

                using (var inPtrs = new MatrixPointersBag<T>(true, inputs.ToArray()))
                using (var targPtrs = new MatrixPointersBag<T>(true, targets.ToArray()))
                {
                    fixed (MatrixDefinition* inPtr = &inPtrs.Definitions[0], targPtr = &targPtrs.Definitions[0])
                    {
                        return GpuInterface.TrainSequence(_gpuNetworkPtr, inPtr, targPtr, inputs.Count);
                    }
                }
            }
#endif

            return base.TrainSequence(inputs, targets);
        }

        public void TransferStateToHost()
        {
#if !CPUONLY
            if (_gpuNetworkPtr == IntPtr.Zero)
            {
                throw new InvalidOperationException("You are not using GPU!");
            }

            foreach (var layer in LayersList)
            {
                layer.TransferWeightsToHost();
            }
#else
            throw new InvalidOperationException("Library was compiled without GPU support!");
#endif
        }

        public void UseGpu()
        {
#if !CPUONLY
            if (_gpuNetworkPtr != IntPtr.Zero)
            {
                return;
            }

            CreateGpuNetwork();
#else
            throw new InvalidOperationException("Library was compiled without GPU support!");
#endif
        }

        public void UseCpu()
        {
#if !CPUONLY
            DestroyGpuNetwork();
#endif
        }

        public override void Save(Stream s)
        {
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_magic);
                writer.Write(BatchSize);
                writer.Write(SeqLen);
                writer.Write(LayersList.Count);

                foreach (var layer in LayersList)
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
            if (_gpuNetworkPtr != IntPtr.Zero)
            {
                GpuInterface.OptimizeNetwork(_gpuNetworkPtr);
                return;
            }
#endif

            foreach (var layer in LayersList)
                layer.Optimize(Optimizer);
        }

        public override double Error(Matrix<T> y, Matrix<T> target)
        {
            return OutLayer.LayerError(y, target);
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> targets, bool needInputSens = false)
        {  
            List<Matrix<T>> prop = OutLayer.ErrorPropagate(targets);
            if (LayersList.Count < 2)
                return prop;
            for (int i = LayersList.Count - 2; i > 0; i--)
            {
                var layer = LayersList[i];
                prop = layer.BackPropagate(prop, true);
            }
            return InLayer.BackPropagate(prop, needInputSens);
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            var prop = input;
            foreach (var layer in LayersList)
                prop = layer.Step(prop, inTraining);
            return prop;
        }

        public override void ResetMemory()
        {
#if !CPUONLY
            if (_gpuNetworkPtr != IntPtr.Zero)
            {
                GpuInterface.ResetNetworkMemory(_gpuNetworkPtr);
                return;
            }
#endif

            foreach (var layer in LayersList)
                layer.ResetMemory();
        }

        public override void ResetOptimizer()
        {
#if !CPUONLY
            if (_gpuNetworkPtr != IntPtr.Zero)
            {
                GpuInterface.ResetOptimizerCaches(_gpuNetworkPtr);
                return;
            }
#endif

            foreach (var layer in LayersList)
                layer.ResetOptimizer();
        }

        public override void InitSequence()
        {
            foreach (var layer in LayersList)
                layer.InitSequence();
        }

        private void SetParam(int i, T value)
        {
            if (i >= TotalParamCount)
                throw new ArgumentException($"Parameter index ({i}) should be less than {TotalParamCount}");

            var paramCnt = 0;
            foreach (var layer in LayersList)
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
            foreach (var layer in LayersList)
            {
                if (i < paramCnt + layer.TotalParamCount)
                    return layer.GetParam(i - paramCnt, grad);
                paramCnt += layer.TotalParamCount;
            }
            throw new Exception($"What the fuck is this? Your index={i} is somehow less then TotalParamCount={TotalParamCount} but more than sum of all layer param counts {paramCnt}!");
        }

        public void Dispose()
        {
#if !CPUONLY
            DestroyGpuNetwork();
#endif
        }

        private void DestroyGpuNetwork()
        {
            if (_gpuNetworkPtr != IntPtr.Zero)
            {
                GpuInterface.DestroyLayeredNetwork(_gpuNetworkPtr);
                _gpuNetworkPtr = IntPtr.Zero;
            }
        }

        public override IReadOnlyList<NeuroWeight<T>> Weights => LayersList.SelectMany(x => x.Weights).ToList();

        private void CreateGpuNetwork()
        {
            DestroyGpuNetwork();

            _gpuNetworkPtr = GpuInterface.CreateLayeredNetwork(InputSize, OutputSize, BatchSize, SeqLen);
            for (int i = 0; i < LayersList.Count; i++)
            {
                var layer = LayersList[i].CreateGpuLayer();
                GpuInterface.AddNetworkLayer(_gpuNetworkPtr, layer);
            }

            var optimizer = _optimizer.CreateGpuOptimizer();
            GpuInterface.SetNetworkOptimizer(_gpuNetworkPtr, optimizer);
        }

        #region Candidates for removal

        #endregion
    }
}