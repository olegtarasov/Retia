using System;
using System.Runtime.InteropServices;

namespace Retia
{
    public static class GpuInterface
    {
        [DllImport(Const.CudaDllName)]
        public static extern IntPtr CreateLayeredNetwork(int inputSize, int outputSize, int batchSize, int seqLen);

        [DllImport(Const.CudaDllName)]
        public static extern void DestroyLayeredNetwork(IntPtr network);

        [DllImport(Const.CudaDllName)]
        public static extern void AddNetworkLayer(IntPtr network, IntPtr layer);

        [DllImport(Const.CudaDllName)]
        public static extern void SetNetworkOptimizer(IntPtr network, IntPtr optimizer);

        [DllImport(Const.CudaDllName)]
        public static extern IntPtr CreateGruLayer(int inputSize, int hSize, int layers, int batchSize, int seqLen);

        [DllImport(Const.CudaDllName)]
        public static extern IntPtr CreateSoftmaxLayer(int inSize, int batchSize, int seqLen);

        [DllImport(Const.CudaDllName)]
        public static extern void TransferLayerStatesFromHost(IntPtr layer, IntPtr matrices, int matrixCount);

        [DllImport(Const.CudaDllName)]
        public static extern void TransferLayerStatesToHost(IntPtr layer, IntPtr matrices, int matrixCount);

        [DllImport(Const.CudaDllName)]
        public static extern void SetLearningRate(IntPtr optimizer, float learningRate);

        [DllImport(Const.CudaDllName)]
        public static extern void DestroyOptimizer(IntPtr optimizer);

        [DllImport(Const.CudaDllName)]
        public static extern IntPtr CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay);
    }
}