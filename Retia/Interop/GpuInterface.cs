using System;
using System.Runtime.InteropServices;

namespace Retia.Interop
{
    public static class GpuInterface
    {
        public const string CudaDllName = "Retia.Cuda.dll";

        public abstract class TestingBase
        {
            public abstract double TestCrossEntropyError(MatrixDefinition m1, MatrixDefinition m2);
            public abstract void TestCrossEntropyBackprop(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result);
            public abstract void TestRMSPropUpdate(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1,
                                                   MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);

            public abstract void TestClampMatrix(MatrixDefinition matrix, float threshold);
            public abstract void TestMatrixTransfer(MatrixDefinition matrix);
            public abstract void TestMatrixTransferRowMajor(MatrixDefinition matrix);

            public abstract void TestWeightTransfer(WeightDefinition weight);
            public abstract void TestWeightTransferRowMajor(WeightDefinition weight);
        }

        public class CpuTesting : TestingBase
        {
            public static TestingBase Instance { get; } = new CpuTesting();

            public override double TestCrossEntropyError(MatrixDefinition m1, MatrixDefinition m2)
            {
                return Testing.TestCrossEntropyErrorCpu(m1, m2);
            }

            public override void TestCrossEntropyBackprop(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result)
            {
                Testing.TestCrossEntropyBackpropCpu(m1, m2, result);
            }

            public override void TestRMSPropUpdate(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1, MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
            {
                Testing.TestRMSPropUpdateCpu(weight, grad, cache1, cache2, cacheM, learningRate, decayRate, momentum, weightDecay);
            }

            public override void TestClampMatrix(MatrixDefinition matrix, float threshold)
            {
                Testing.TestClampMatrixCpu(matrix, threshold);
            }

            public override void TestMatrixTransfer(MatrixDefinition matrix)
            {
                Testing.TestMatrixTransferCpu(matrix);
            }

            public override void TestMatrixTransferRowMajor(MatrixDefinition matrix)
            {
                Testing.TestMatrixTransferRowMajorCpu(matrix);
            }

            public override void TestWeightTransfer(WeightDefinition weight)
            {
                Testing.TestWeightTransferCpu(weight);
            }

            public override void TestWeightTransferRowMajor(WeightDefinition weight)
            {
                Testing.TestWeightTransferRowMajorCpu(weight);
            }
        }

        public class GpuTesting : TestingBase
        {
            public static TestingBase Instance { get; } = new GpuTesting();

            public override double TestCrossEntropyError(MatrixDefinition m1, MatrixDefinition m2)
            {
                return Testing.TestCrossEntropyErrorGpu(m1, m2);
            }

            public override void TestCrossEntropyBackprop(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result)
            {
                Testing.TestCrossEntropyBackpropGpu(m1, m2, result);
            }

            public override void TestRMSPropUpdate(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1, MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
            {
                Testing.TestRMSPropUpdateGpu(weight, grad, cache1, cache2, cacheM, learningRate, decayRate, momentum, weightDecay);
            }

            public override void TestClampMatrix(MatrixDefinition matrix, float threshold)
            {
                Testing.TestClampMatrixGpu(matrix, threshold);
            }

            public override void TestMatrixTransfer(MatrixDefinition matrix)
            {
                Testing.TestMatrixTransferGpu(matrix);
            }

            public override void TestMatrixTransferRowMajor(MatrixDefinition matrix)
            {
                Testing.TestMatrixTransferRowMajorGpu(matrix);
            }

            public override void TestWeightTransfer(WeightDefinition weight)
            {
                Testing.TestWeightTransferGpu(weight);
            }

            public override void TestWeightTransferRowMajor(WeightDefinition weight)
            {
                Testing.TestWeightTransferRowMajorGpu(weight);
            }
        }

        public static class Testing
        {
            [DllImport(CudaDllName)]
            public static extern double TestCrossEntropyErrorCpu(MatrixDefinition m1, MatrixDefinition m2);

            [DllImport(CudaDllName)]
            public static extern double TestCrossEntropyErrorGpu(MatrixDefinition m1, MatrixDefinition m2);

            [DllImport(CudaDllName)]
            public static extern void TestCrossEntropyBackpropCpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result);

            [DllImport(CudaDllName)]
            public static extern void TestCrossEntropyBackpropGpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result);

            [DllImport(CudaDllName)]
            public static extern void TestRMSPropUpdateCpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1,
                                                            MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);

            [DllImport(CudaDllName)]
            public static extern void TestRMSPropUpdateGpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1,
                                                            MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);

            [DllImport(CudaDllName)]
            public static extern void TestClampMatrixCpu(MatrixDefinition matrix, float threshold);

            [DllImport(CudaDllName)]
            public static extern void TestClampMatrixGpu(MatrixDefinition matrix, float threshold);

            [DllImport(CudaDllName)]
            public static extern void TestMatrixTransferCpu(MatrixDefinition matrix);

            [DllImport(CudaDllName)]
            public static extern void TestMatrixTransferGpu(MatrixDefinition matrix);

            [DllImport(CudaDllName)]
            public static extern void TestMatrixTransferRowMajorCpu(MatrixDefinition matrix);

            [DllImport(CudaDllName)]
            public static extern void TestMatrixTransferRowMajorGpu(MatrixDefinition matrix);

            [DllImport(CudaDllName)]
            public static extern void TestWeightTransferCpu(WeightDefinition weight);

            [DllImport(CudaDllName)]
            public static extern void TestWeightTransferGpu(WeightDefinition weight);

            [DllImport(CudaDllName)]
            public static extern void TestWeightTransferRowMajorCpu(WeightDefinition weight);

            [DllImport(CudaDllName)]
            public static extern void TestWeightTransferRowMajorGpu(WeightDefinition weight);
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
        public static extern void OptimizeNetwork(IntPtr network);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateGruLayer(int inputSize, int hSize, int layers, int batchSize, int seqLen);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateSoftmaxLayer(int inSize, int batchSize, int seqLen);

        [DllImport(CudaDllName)]
        public static extern unsafe void TransferLayerStatesToDevice(IntPtr layer, WeightDefinition *weigths, int count);

        [DllImport(CudaDllName)]
        public static extern unsafe void TransferLayerStatesToHost(IntPtr layer, WeightDefinition *weights, int count);

        [DllImport(CudaDllName)]
        public static extern void ResetNetworkMemory(IntPtr network);

        [DllImport(CudaDllName)]
        public static extern void ResetOptimizerCaches(IntPtr network);

        [DllImport(CudaDllName)]
        public static extern void SetLearningRate(IntPtr optimizer, float learningRate);

        [DllImport(CudaDllName)]
        public static extern void DestroyOptimizer(IntPtr optimizer);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay);

        [DllImport(CudaDllName)]
        public static extern unsafe double TrainSequence(IntPtr net, MatrixDefinition *inputs, MatrixDefinition *targets, int count);

        [DllImport(CudaDllName)]
        public static extern IntPtr CreateLinearLayer(int inputSize, int outSize, int batchSize, int seqLen);

        [DllImport(CudaDllName)]
        public static extern unsafe void TransferLayerOutputsToHost(IntPtr layer, MatrixDefinition* outputs, int count);

        [DllImport(CudaDllName)]
        public static extern unsafe void LayerForwardSequence(IntPtr layer, MatrixDefinition* inputs, int count);
    }
}