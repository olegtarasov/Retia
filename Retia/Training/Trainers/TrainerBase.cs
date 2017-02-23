using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Retia.Integration;
using Retia.Neural;
using Retia.Training.Data;
using Retia.Training.Testers;
using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
    public abstract class TrainerBase<T, TOptions, TReport> : ITrainingReporter<TReport>, ITrainerEvents 
        where T : struct, IEquatable<T>, IFormattable
        where TOptions : TrainerOptionsBase
        where TReport : TrainReportEventArgsBase
    {
        public TOptions Options { get; }

        private readonly ManualResetEventSlim _pauseHandle = new ManualResetEventSlim(true);
        
        protected TrainerBase(TOptions options)
        {
            Options = options;

            ValidateOptions(options);
        }

        public abstract NeuralNet<T> TestableNetwork { get; }

        public List<PeriodicActionBase> PeriodicActions { get; } = new List<PeriodicActionBase>();
        public bool IsTraining { get; private set; }
        public bool IsPaused { get; private set; }
        public long Epoch { get; protected set; }
        
        public long Iteration { get; protected set; }

        protected abstract TReport GetTrainingReport();

        protected abstract void TrainIteration();

        protected abstract void ResetMemory();

        public event EventHandler<TReport> TrainReport;

        public void Pause()
        {
            IsPaused = true;
            _pauseHandle.Reset();
        }

        public void Resume()
        {
            IsPaused = false;
            _pauseHandle.Set();
        }

        public async Task Train(CancellationToken token)
        {
            if (IsTraining)
            {
                throw new InvalidOperationException("Already training!");
            }

            IsTraining = true;
            IsPaused = false;

            try
            {
                await Task.Run(() => TrainInternal(token), token);
            }
            catch (Exception e)
            {
                Options.ProgressWriter?.Message(e.ToString());
            }
        }

        public event Action SequenceTrained;
        public event Action EpochReached;

        protected virtual void InitTraining()
        {
        }

        protected virtual string GetTrainingReportMessage()
        {
            return null;
        }

        protected virtual void ValidateOptions(TOptions options)
        {
        }

        protected void OnTrainReport(TReport args)
        {
            TrainReport?.Invoke(this, args);
        }

        protected void OnEpochReached()
        {
            EpochReached?.Invoke();
        }

        private void TrainInternal(CancellationToken token)
        {
            Iteration = 0;

            InitTraining();
            SubscribeActions();

            var watch = new Stopwatch();
            watch.Start();
            while (IsTraining)
            {
                if (token.IsCancellationRequested)
                {
                    Options.ProgressWriter?.Message("Stopped training manually");
                    IsTraining = false;
                    return;
                }

                if (!_pauseHandle.IsSet)
                {
                    Options.ProgressWriter?.Message("Training paused");
                    try
                    {
                        _pauseHandle.Wait(token);
                    }
                    catch (TaskCanceledException)
                    {
                        return;
                    }
                    Options.ProgressWriter?.Message("Training resumed");
                }

                watch.Restart();
                TrainIteration();
                watch.Stop();

                Iteration++;

                // Check for memory reset on iteration.
                if (Options.ResetMemory.ShouldDoOnIteration(Iteration))
                {
                    ResetMemory();
                }

                OnSequenceTrained();

                if (Options.ReportProgress.ShouldDoOnIteration(Iteration))
                {
                    OnTrainReport(GetTrainingReport());

                    if (Options.ReportMesages && Options.ProgressWriter != null)
                    {
                        var preIter = new StringBuilder();
                        preIter.Append('#').Append(Epoch).Append('[');

                        var postIter = new StringBuilder();
                        postIter.Append(' ').AppendFormat("{0:0.0000}", watch.Elapsed.TotalSeconds).Append("s] ").Append(GetTrainingReportMessage());

                        preIter.Append(GetIterationProgress(preIter.Length + postIter.Length)).Append(postIter);

                        Options.ProgressWriter.SetItemProgress(preIter.ToString());
                    }
                }

                if (Epoch > Options.MaxEpoch)
                {
                    Options.ProgressWriter?.Message($"{Options.MaxEpoch} reached, stopped training.");
                    return;
                }
            }

            UnsubscribeActions();
        }

        protected virtual string GetIterationProgress(int otherLen)
        {
            return $"I:{Iteration}";
        }

        protected virtual void SubscribeActions()
        {
            foreach (var action in PeriodicActions)
            {
                action.Subscribe(this);
            }
        }

        protected virtual void UnsubscribeActions()
        {
            foreach (var action in PeriodicActions)
            {
                action.Unsubscribe();
            }
        }

        private void OnSequenceTrained()
        {
            SequenceTrained?.Invoke();
        }
    }
}