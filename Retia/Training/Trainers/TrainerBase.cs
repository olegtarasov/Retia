using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Retia.Neural;
using Retia.Training.Data;
using Retia.Training.Testers;
using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
    public abstract class TrainerBase<T, TOptions, TReport> : ITrainingReporter<TReport>, ITrainerEvents where T : struct, IEquatable<T>, IFormattable
        where TOptions : TrainerOptionsBase
        where TReport : TrainReportEventArgsBase
    {
        protected readonly IDataProvider<T> DataProvider;
        public TOptions Options { get; }

        private readonly ITester<T> _tester;
        private readonly ManualResetEventSlim _pauseHandle = new ManualResetEventSlim(true);
        
        protected TrainerBase(IDataProvider<T> dataProvider, ITester<T> tester, TOptions options)
        {
            DataProvider = dataProvider;
            Options = options;

            _tester = tester;

            ValidateOptions(options);

            DataProvider.TrainingSetChanged += DataProviderOnTrainingSetChanged;
            DataProvider.TestSetChanged += DataProviderOnTestSetChanged;
        }

        public abstract NeuralNet<T> TestableNetwork { get; }

        public ITrainingStatusWriter StatusWriter { get; set; }
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
                StatusWriter?.Message(e.ToString());
            }
        }

        public event Action SequenceTrained;
        public event Action EpochReached;

        protected virtual void DataProviderOnTestSetChanged(object sender, DataSetChangedArgs<T> dataSetChangedArgs)
        {
        }

        protected virtual void DataProviderOnTrainingSetChanged(object sender, DataSetChangedArgs<T> e)
        {
        }

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
            var testWatch = new Stopwatch();
            
            watch.Start();
            while (IsTraining)
            {
                if (token.IsCancellationRequested)
                {
                    StatusWriter?.Message("Stopped training manually");
                    IsTraining = false;
                    return;
                }

                if (!_pauseHandle.IsSet)
                {
                    StatusWriter?.Message("Training paused");
                    try
                    {
                        _pauseHandle.Wait(token);
                    }
                    catch (TaskCanceledException)
                    {
                        return;
                    }
                    StatusWriter?.Message("Training resumed");
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

                    if (Options.ReportMesages)
                    {
                        string progress = $"#{Epoch}[{GetIterationProgress()} {watch.Elapsed.TotalSeconds:0.0000}s] {GetTrainingReportMessage()}";
                        StatusWriter?.UpdateEpochStatus(progress);
                    }
                }

                if (_tester != null && Options.RunTests.ShouldDoOnIteration(Iteration))
                {
                    if (DataProvider.TestSet == null)
                    {
                        throw new InvalidOperationException("Test set was not created!");
                    }

                    DataProvider.TestSet.Reset();
                    testWatch.Restart();
                    var result = _tester.Test(TestableNetwork.Clone(), DataProvider.TestSet);
                    testWatch.Stop();

                    if (Options.ReportMesages)
                    {
                        StatusWriter?.Message($"=========\n\tTested in:\t{testWatch.Elapsed.TotalSeconds:0.0000}s\n{result.GetReport()}\n=========\n");
                    }
                }

                if (Epoch > Options.MaxEpoch)
                {
                    StatusWriter?.Message($"{Options.MaxEpoch} reached, stopped training.");
                    return;
                }
            }

            UnsubscribeActions();
        }

        protected virtual string GetIterationProgress()
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