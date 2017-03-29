using System;
using System.Linq;
using CLAP;
using Retia.Gui;
using Retia.Gui.Models;
using Retia.Gui.Windows;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.ErrorFunctions;
using Retia.Neural.Layers;
using Retia.Optimizers;
using Retia.Training.Data.Samples;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;
using Retia.Training.Trainers.Sessions;

namespace SimpleExamples
{
    public partial class Examples
    {
        [Verb]
        public void Xor()
        {
            MklProvider.TryUseMkl(true, ConsoleProgressWriter.Instance);

            var optimizer = new RMSPropOptimizer<float>(1e-3f);
            var net = new LayeredNet<float>(1, 1, new AffineLayer<float>(2, 3, AffineActivation.Tanh), new AffineLayer<float>(3, 1, AffineActivation.Tanh) { ErrorFunction = new MeanSquareError<float>() })
            {
                Optimizer = optimizer
            };

            var trainer = new OptimizingTrainer<float>(net, optimizer, new XorDataset(true), new OptimizingTrainerOptions(1)
            {
                ErrorFilterSize = 0,
                ReportProgress = new EachIteration(1),
                ReportMesages = true,
                ProgressWriter = ConsoleProgressWriter.Instance,
                LearningRateScaler = new ProportionalLearningRateScaler(new EachIteration(1), 9e-5f)
            }, new OptimizingSession("XOR"));

            var runner = ConsoleRunner.Create(trainer, net);

            trainer.TrainReport += (sender, args) =>
            {
                if (args.Errors.Last().RawError < 1e-7f)
                {
                    runner.Stop();
                    Console.WriteLine("Finished training.");
                }
            };

            var gui = new RetiaGui();
            gui.RunAsync(() => new TrainingWindow(new TypedTrainingModel<float>(trainer)));

            runner.Run();
        }

    }
}