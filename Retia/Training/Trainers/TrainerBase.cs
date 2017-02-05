using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Retia.Integration;
using Retia.Neural;
using Retia.Training.Data;
using Retia.Training.Testers;

namespace Retia.Training.Trainers
{
    public abstract class TrainerBase<TOptions, TReport> : ILogReporter, ITrainingReporter<TReport> 
        where TOptions : TrainerOptionsBase
        where TReport : TrainReportEventArgsBase
    {
        protected readonly IDataProvider DataProvider;
        protected readonly TOptions Options;

        private readonly ITester _tester;
        private readonly ManualResetEventSlim _pauseHandle = new ManualResetEventSlim(true);


        protected TrainerBase(IDataProvider dataProvider, ITester tester, TOptions options)
        {
            DataProvider = dataProvider;
            Options = options;

            _tester = tester;

            ValidateOptions(options);

            DataProvider.TrainingSetChanged += DataProviderOnTrainingSetChanged;
            DataProvider.TestSetChanged += DataProviderOnTestSetChanged;
        }

        public abstract NeuralNet TestableNetwork { get; }

        public bool IsTraining { get; private set; }
        public long Epoch { get; protected set; }
        
        public long Iteration { get; protected set; }

        protected abstract TReport GetTrainingReport(bool userTest);

        protected abstract void TrainIteration();

        protected abstract void ResetMemory();

        public event EventHandler<LogEventArgs> Message;
        public event EventHandler<TReport> TrainReport;

        public void Pause()
        {
            _pauseHandle.Reset();
        }

        public void Resume()
        {
            _pauseHandle.Set();
        }

        public async Task Train(CancellationToken token)
        {
            if (IsTraining)
            {
                throw new InvalidOperationException("Already training!");
            }

            IsTraining = true;

            try
            {
                await Task.Run(() => TrainInternal(token), token);
            }
            catch (Exception e)
            {
                OnMessage(e.ToString());
            }
        }

        public event EventHandler<TReport> UserTest;
        public event Action SequenceTrained;
        public event Action EpochReached;

        protected virtual void DataProviderOnTestSetChanged(object sender, DataSetChangedArgs dataSetChangedArgs)
        {
        }

        protected virtual void DataProviderOnTrainingSetChanged(object sender, DataSetChangedArgs e)
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

        protected virtual void OnUserTest(TReport e)
        {
            UserTest?.Invoke(this, e);
        }

        protected void OnMessage(string message)
        {
            Message?.Invoke(this, new LogEventArgs(message));
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

            var watch = new Stopwatch();
            var testWatch = new Stopwatch();
            
            watch.Start();
            while (IsTraining)
            {
                if (token.IsCancellationRequested)
                {
                    OnMessage("Stopped training manually");
                    IsTraining = false;
                    return;
                }

                if (!_pauseHandle.IsSet)
                {
                    OnMessage("Training paused");
                    try
                    {
                        _pauseHandle.Wait(token);
                    }
                    catch (TaskCanceledException)
                    {
                        return;
                    }
                    OnMessage("Training resumed");
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
                    OnTrainReport(GetTrainingReport(false));

                    if (Options.ReportMesages)
                    {
                        string progress = $"Epoch #{Epoch:F3} - iter #{Iteration}:";

                        OnMessage(progress);
                        OnMessage("---------");

                        string additionalReport = GetTrainingReportMessage();
                        if (!string.IsNullOrEmpty(additionalReport))
                        {
                            OnMessage(additionalReport);
                        }

                        OnMessage($"\tDuration:\t{watch.Elapsed.TotalSeconds:0.0000}s");
                        OnMessage("---------\n");
                    }
                }

                if (Options.RunUserTests.ShouldDoOnIteration(Iteration))
                {
                    OnUserTest(GetTrainingReport(true));
                }

                if (_tester != null && Options.RunTests.ShouldDoOnIteration(Iteration))
                {
                    if (DataProvider.TestSet == null)
                    {
                        OnMessage("=========");
                        OnMessage("Trying to run network on test set, but test data set was not created!");
                        OnMessage("=========\n");
                    }
                    else
                    {
                        DataProvider.TestSet.Reset();
                        testWatch.Restart();
                        var result = _tester.Test(TestableNetwork.Clone(), DataProvider.TestSet);
                        testWatch.Stop();

                        if (Options.ReportMesages)
                        {
                            OnMessage("=========");
                            OnMessage($"\tTested in:\t{testWatch.Elapsed.TotalSeconds:0.0000}s");
                            OnMessage(result.GetReport());
                            OnMessage("=========\n");
                        }
                    }
                }

                if (Epoch > Options.MaxEpoch)
                {
                    OnMessage($"{Options.MaxEpoch} reached, stopped training.");
                    return;
                }
            }
        }


        private void OnSequenceTrained()
        {
            SequenceTrained?.Invoke();
        }
    }
}