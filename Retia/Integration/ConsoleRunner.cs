using System;
using System.Threading;
using System.Threading.Tasks;
using Retia.Neural;
using Retia.Training.Trainers;

namespace Retia.Integration
{
    /// <summary>
    /// Runs a trainer on a console supporting cancellation and other options.
    /// </summary>
    public static class ConsoleRunner
    {
        public static void RunTrainer<T, TOptions, TReport>(TrainerBase<T, TOptions, TReport> trainer, NeuralNet<T> network)
            where T : struct, IEquatable<T>, IFormattable
            where TOptions : TrainerOptionsBase
            where TReport : TrainReportEventArgsBase
        {
            var cts = new CancellationTokenSource();
            var task = trainer.Train(cts.Token);
            var running = true;
            while (running)
            {
                var c = Console.ReadKey().Key;
                switch (c)
                {
                    case ConsoleKey.Q:
                        running = false;
                        break;

                    case ConsoleKey.R:
                        Console.WriteLine("Reseting optimizer cache!");
                        trainer.Pause();
                        network.ResetOptimizer();
                        trainer.Resume();
                        break;
                }
            }

            cts.Cancel();
            Task.WaitAll(task);
        }
    }
}