#if !CPUONLY
using System;
using System.Linq;
using System.Threading;
using Retia.Integration;
using Retia.Interop;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.ErrorFunctions;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.Tests.Interop;
using Retia.Tests.Plumbing;
using Retia.Training.Data.Samples;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;
using Retia.Training.Trainers.Sessions;
using Xunit;

namespace Retia.Tests.Gpu
{
    public class GpuTransferTests : TransferTestsBase
    {
        protected override GpuInterface.TestingBase Interface => GpuInterface.GpuTesting.Instance;

        [Fact]
        public void CanTransferWeightThroughNeuroWeight()
        {
            var weight = MatrixFactory.RandomMatrix<float>(2, 3, 5.0f);
            var local = new NeuroWeight<float>(weight);
            var remote = local.Clone();

            MutateMatrix(local.Weight);
            MutateMatrix(local.Gradient);
            MutateMatrix(local.Cache1);
            MutateMatrix(local.Cache2);
            MutateMatrix(local.CacheM);

            using (var ptrs = new WeightDefinitionBag<float>(remote))
            {
                GpuInterface.Testing.TestComplexWeightTransfer(ptrs.Definitions[0]);
            }

            local.Weight.ShouldMatrixEqualWithinError(remote.Weight);
            local.Gradient.ShouldMatrixEqualWithinError(remote.Gradient);
            local.Cache1.ShouldMatrixEqualWithinError(remote.Cache1);
            local.Cache2.ShouldMatrixEqualWithinError(remote.Cache2);
            local.CacheM.ShouldMatrixEqualWithinError(remote.CacheM);
        }

        [Fact]
        public void CanTransferWeightThroughNeuroWeightRowMajor()
        {
            var weight = MatrixFactory.RandomMatrix<float>(2, 3, 5.0f);
            var local = new NeuroWeight<float>(weight);
            var remote = local.Clone();

            MutateMatrixRowMajor(local.Weight);
            MutateMatrixRowMajor(local.Gradient);
            MutateMatrixRowMajor(local.Cache1);
            MutateMatrixRowMajor(local.Cache2);
            MutateMatrixRowMajor(local.CacheM);

            using (var ptrs = new WeightDefinitionBag<float>(true, remote))
            {
                GpuInterface.Testing.TestComplexWeightTransferRowMajor(ptrs.Definitions[0]);
            }

            local.Weight.ShouldMatrixEqualWithinError(remote.Weight);
            local.Gradient.ShouldMatrixEqualWithinError(remote.Gradient);
            local.Cache1.ShouldMatrixEqualWithinError(remote.Cache1);
            local.Cache2.ShouldMatrixEqualWithinError(remote.Cache2);
            local.CacheM.ShouldMatrixEqualWithinError(remote.CacheM);
        }

        /// <summary>
        /// There is no point in teaching XOR problem to GRU network, but in this test
        /// we don't care about actual learning, only in weight transfer.
        /// </summary>
        [Fact]
        public void CanTransferWeightsInsideNetwrok()
        {
            var o1 = new RMSPropOptimizer<float>(1e-3f);
            var n1 = new LayeredNet<float>(1, 1, 
                new GruLayer<float>(2, 3),
                new LinearLayer<float>(3, 2),
                new SoftMaxLayer<float>(2))
            {
                Optimizer = o1
            };

            var n2 = (LayeredNet<float>)n1.Clone();
            var o2 = new RMSPropOptimizer<float>(1e-3f);
            n2.Optimizer = o2;

            n1.UseGpu();
            TrainXor(n1);
            n1.TransferStateToHost();

            TrainXor(n2);

            var w1 = n1.Weights.ToList();
            var w2 = n2.Weights.ToList();

            for (int i = 0; i < w1.Count; i++)
            {
                w1[i].Weight.ShouldMatrixEqualWithinError(w2[i].Weight);
                w1[i].Gradient.ShouldMatrixEqualWithinError(w2[i].Gradient);
                w1[i].Cache1.ShouldMatrixEqualWithinError(w2[i].Cache1);
                w1[i].Cache2.ShouldMatrixEqualWithinError(w2[i].Cache2);
                w1[i].CacheM.ShouldMatrixEqualWithinError(w2[i].CacheM);
            }
        }

        private void TrainXor(LayeredNet<float> net)
        {
            var trainer = new OptimizingTrainer<float>(net, net.Optimizer, new XorDataset(true), new OptimizingTrainerOptions(1)
            {
                ErrorFilterSize = 0,
                ReportProgress = ActionSchedule.Disabled,
                ReportMesages = false
            }, new OptimizingSession(false));

            var cts = new CancellationTokenSource();
            trainer.SequenceTrained += s =>
            {
                cts.Cancel();
            };

            trainer.Train(cts.Token).Wait();
        }
    }
}

#endif