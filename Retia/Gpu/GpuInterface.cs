using System;
using System.Runtime.InteropServices;

namespace Retia.Gpu
{
    public static class GpuInterface
    {
        public const string CudaDllName = "Retia.Cuda.dll";

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateLayeredNetwork(int inputSize, int outputSize, int batchSize, int seqLen);

        [DllImport(CudaDllName)]
        public static extern void DestroyLayeredNetwork(IntPtr network);

        [DllImport(CudaDllName)]
        public static extern void AddNetworkLayer(IntPtr network, IntPtr layer);

        [DllImport(CudaDllName)]
        public static extern void SetNetworkOptimizer(IntPtr network, IntPtr optimizer);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateGruLayer(int inputSize, int hSize, int layers, int batchSize, int seqLen);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateSoftmaxLayer(int inSize, int batchSize, int seqLen);

        [DllImport(CudaDllName)]
        public static extern unsafe void TransferLayerStatesFromHost(IntPtr layer, HostMatrixDefinition *matrices, int matrixCount);

        [DllImport(CudaDllName)]
        public static extern unsafe void TransferLayerStatesToHost(IntPtr layer, HostMatrixDefinition *matrices, int matrixCount);

        [DllImport(CudaDllName)]
        public static extern void SetLearningRate(IntPtr optimizer, float learningRate);

        [DllImport(CudaDllName)]
        public static extern void DestroyOptimizer(IntPtr optimizer);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay);

        [DllImport(CudaDllName)]
        public static extern unsafe double TrainSequence(IntPtr net, HostMatrixDefinition *inputs, HostMatrixDefinition *targets, int count);
    }
}