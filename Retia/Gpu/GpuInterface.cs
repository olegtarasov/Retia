using System;
using System.Runtime.InteropServices;

namespace Retia.Gpu
{
    public static class GpuInterface
    {
        public const string CudaDllName = "Retia.Cuda.dll";

        public abstract class TestingBase
        {
            public abstract double TestCrossEntropyError(HostMatrixDefinition m1, HostMatrixDefinition m2);
            public abstract void TestCrossEntropyBackprop(HostMatrixDefinition m1, HostMatrixDefinition m2, HostMatrixDefinition result);
            public abstract void TestRMSPropUpdate(HostMatrixDefinition weight, HostMatrixDefinition grad, HostMatrixDefinition cache1,
                                                   HostMatrixDefinition cache2, HostMatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);

            public abstract void TestClampMatrix(HostMatrixDefinition matrix, float threshold);
            public abstract void TestMatrixTransfer(HostMatrixDefinition matrix);
        }

        public class CpuTesting : TestingBase
        {
            public static TestingBase Instance { get; } = new CpuTesting();

            public override double TestCrossEntropyError(HostMatrixDefinition m1, HostMatrixDefinition m2)
            {
                return Testing.TestCrossEntropyErrorCpu(m1, m2);
            }

            public override void TestCrossEntropyBackprop(HostMatrixDefinition m1, HostMatrixDefinition m2, HostMatrixDefinition result)
            {
                Testing.TestCrossEntropyBackpropCpu(m1, m2, result);
            }

            public override void TestRMSPropUpdate(HostMatrixDefinition weight, HostMatrixDefinition grad, HostMatrixDefinition cache1, HostMatrixDefinition cache2, HostMatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
            {
                Testing.TestRMSPropUpdateCpu(weight, grad, cache1, cache2, cacheM, learningRate, decayRate, momentum, weightDecay);
            }

            public override void TestClampMatrix(HostMatrixDefinition matrix, float threshold)
            {
                Testing.TestClampMatrixCpu(matrix, threshold);
            }

            public override void TestMatrixTransfer(HostMatrixDefinition matrix)
            {
                Testing.TestMatrixTransferCpu(matrix);
            }
        }

        public class GpuTesting : TestingBase
        {
            public static TestingBase Instance { get; } = new GpuTesting();

            public override double TestCrossEntropyError(HostMatrixDefinition m1, HostMatrixDefinition m2)
            {
                return Testing.TestCrossEntropyErrorGpu(m1, m2);
            }

            public override void TestCrossEntropyBackprop(HostMatrixDefinition m1, HostMatrixDefinition m2, HostMatrixDefinition result)
            {
                Testing.TestCrossEntropyBackpropGpu(m1, m2, result);
            }

            public override void TestRMSPropUpdate(HostMatrixDefinition weight, HostMatrixDefinition grad, HostMatrixDefinition cache1, HostMatrixDefinition cache2, HostMatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
            {
                Testing.TestRMSPropUpdateGpu(weight, grad, cache1, cache2, cacheM, learningRate, decayRate, momentum, weightDecay);
            }

            public override void TestClampMatrix(HostMatrixDefinition matrix, float threshold)
            {
                Testing.TestClampMatrixGpu(matrix, threshold);
            }

            public override void TestMatrixTransfer(HostMatrixDefinition matrix)
            {
                Testing.TestMatrixTransferGpu(matrix);
            }
        }

        public static class Testing
        {
            [DllImport(CudaDllName)]
            public static extern double TestCrossEntropyErrorCpu(HostMatrixDefinition m1, HostMatrixDefinition m2);

            [DllImport(CudaDllName)]
            public static extern double TestCrossEntropyErrorGpu(HostMatrixDefinition m1, HostMatrixDefinition m2);

            [DllImport(CudaDllName)]
            public static extern void TestCrossEntropyBackpropCpu(HostMatrixDefinition m1, HostMatrixDefinition m2, HostMatrixDefinition result);

            [DllImport(CudaDllName)]
            public static extern void TestCrossEntropyBackpropGpu(HostMatrixDefinition m1, HostMatrixDefinition m2, HostMatrixDefinition result);

            [DllImport(CudaDllName)]
            public static extern void TestRMSPropUpdateCpu(HostMatrixDefinition weight, HostMatrixDefinition grad, HostMatrixDefinition cache1,
                                                            HostMatrixDefinition cache2, HostMatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);

            [DllImport(CudaDllName)]
            public static extern void TestRMSPropUpdateGpu(HostMatrixDefinition weight, HostMatrixDefinition grad, HostMatrixDefinition cache1,
                                                            HostMatrixDefinition cache2, HostMatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);

            [DllImport(CudaDllName)]
            public static extern void TestClampMatrixCpu(HostMatrixDefinition matrix, float threshold);

            [DllImport(CudaDllName)]
            public static extern void TestClampMatrixGpu(HostMatrixDefinition matrix, float threshold);

            [DllImport(CudaDllName)]
            public static extern void TestMatrixTransferCpu(HostMatrixDefinition matrix);

            [DllImport(CudaDllName)]
            public static extern void TestMatrixTransferGpu(HostMatrixDefinition matrix);
        }

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
        public static extern unsafe void TransferLayerStatesToDevice(IntPtr layer, HostWeightDefinition *weigths, int count);

        [DllImport(CudaDllName)]
        public static extern unsafe void TransferLayerStatesToHost(IntPtr layer, HostWeightDefinition *weights, int count);

        [DllImport(CudaDllName)]
        public static extern void SetLearningRate(IntPtr optimizer, float learningRate);

        [DllImport(CudaDllName)]
        public static extern void DestroyOptimizer(IntPtr optimizer);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay);

        [DllImport(CudaDllName)]
        public static extern unsafe double TrainSequence(IntPtr net, HostMatrixDefinition *inputs, HostMatrixDefinition *targets, int count);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateLinearLayer(int inputSize, int outSize, int batchSize, int seqLen);
    }
}