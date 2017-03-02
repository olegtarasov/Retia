using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Optimizers;
using Retia.Training.Data;
using Retia.Training.Testers;

namespace Retia.Training.Trainers
{
    public class OptimizingTrainer<T>: TrainerBase<T, OptimizingTrainerOptions, OptimizationReportEventArgs> where T : struct, IEquatable<T>, IFormattable
    {
        protected readonly NeuralNet<T> _network;
        private readonly OptimizerBase<T> _optimizer;
        private readonly ITester<T> _tester;

        private List<double> _errors;
        private MAV _mav;
        private double _lastError;

        private double _dErr = 0;
        private int _processedSamples = 0;
        private IDataSet<T> _trainingSet;

        public OptimizingTrainer(NeuralNet<T> network, OptimizerBase<T> optimizer, ITester<T> tester, OptimizingTrainerOptions options) : base(options)
        {
            _network = network;
            _optimizer = optimizer;
            _tester = tester;
        }

        public override NeuralNet<T> TestableNetwork => _network;

        public IDataSet<T> TrainingSet
        {
            get
            {
                return _trainingSet;
            }
            set
            {
                if (_trainingSet != null)
                    _trainingSet.DataSetReset -= TrainingSetOnDataSetReset;

                _trainingSet = value;
                _trainingSet.DataSetReset += TrainingSetOnDataSetReset;
            }
        }

        public IDataSet<T> TestSet { get; set; }

        // TODO: Find a better way to set learning rate manually
        public float LearningRate
        {
            get
            {
                return _optimizer.LearningRate;
            }
            set
            {
                _optimizer.LearningRate = value;
                Options.LearningRateScaler.Reset();
            }
        }

        protected virtual TrainingSequence<T> GetTrainSamples()
        {
            // We can get TrainingSetOnDataSetReset during this call
            TrainingSequence<T> result;

            result = TrainingSet.GetNextSamples(Options.SequenceLength);

            if (result == null)
            {
                // A new epoch has come, try to get the sequence again
                result = TrainingSet.GetNextSamples(Options.SequenceLength);

                // If we couldn't get a sequence at the start of the epoch, this is a _bug.
                if (result == null)
                {
                    throw new InvalidOperationException("Couldn't get a training sequence at the start of an epoch!");
                }
            }

            return result;
        }

        protected virtual void ProcessError(double error)
        {
            double filteredError;

            if (Options.ErrorFilterSize > 0)
            {
                if (_mav == null)
                {
                    _mav = new MAV(Options.ErrorFilterSize);
                }
                else if (_mav.Order != Options.ErrorFilterSize)
                {
                    _mav.Order = Options.ErrorFilterSize;
                }

                filteredError = _mav.Filter(error);
            }
            else
            {
                if (_mav != null)
                {
                    _mav = null;
                }

                filteredError = error;
            }

            _errors.Add(filteredError);
            _dErr = filteredError - _lastError;
            _lastError = filteredError;
        }

        protected override string GetIterationProgress(int otherLen)
        {
            if (TrainingSet.SampleCount > 0)
            {
                var sb = new StringBuilder();

                int total = TrainingSet.SampleCount / Options.SequenceLength;

                sb.Append(Iteration).Append('/').Append(total);
                sb.Append(ConsoleProgressWriter.GetProgressbar(Iteration, total, otherLen + sb.Length));

                return sb.ToString();
            }

            return base.GetIterationProgress(otherLen);
        }

        protected override void InitTraining()
        {
            base.InitTraining();

            if (TrainingSet == null)
                throw new InvalidOperationException("Training set is not set!");

            if (Options.RunTests?.IsEnabled == true && TestSet == null)
                throw new InvalidOperationException("Tests are enabled, but test set is not set!");

            _errors = new List<double>();
            _mav = Options.ErrorFilterSize > 0 ? new MAV(Options.ErrorFilterSize) : null;
            _lastError = 0;
            _processedSamples = 0;

            Options.ProgressWriter?.Message($"Sequence length: {Options.SequenceLength}");
            Options.ProgressWriter?.Message($"Using network with total param count {_network.TotalParamCount}");
        }

        protected override OptimizationReportEventArgs GetTrainingReport()
        {
            var result = new OptimizationReportEventArgs(_errors.ToList(), Epoch, Iteration, _optimizer.LearningRate);
            _errors.Clear();

            return result;
        }

        protected override void ValidateOptions(OptimizingTrainerOptions options)
        {
            base.ValidateOptions(options);

            if (options.SequenceLength <= 0)
            {
                throw new InvalidOperationException("Invalid sequence length!");
            }
        }

        protected override void TrainIteration()
        {
            var sequence = GetTrainSamples();
            double error = _network.TrainSequence(sequence.Inputs, sequence.Targets);

            _processedSamples += sequence.Inputs.Count;

            _network.Optimize();
            ProcessError(error);

            if (_tester != null && Options.RunTests?.ShouldDoOnIteration(Iteration) == true)
            {
                RunTest();
            }
        }

        protected override void ResetMemory()
        {
            _network.ResetMemory();
        }

        protected override string GetTrainingReportMessage()
        {
            return $"[E:{_lastError:0.0000} | dE:{_dErr:0.0000} | LR:{_optimizer.LearningRate}]";
        }

        protected override void SubscribeActions()
        {
            base.SubscribeActions();
            Options.LearningRateScaler?.Subscribe(this);
        }

        protected override void UnsubscribeActions()
        {
            base.UnsubscribeActions();
            Options.LearningRateScaler?.Unsubscribe();
        }

        private void RunTest()
        {
            if (TestSet == null)
            {
                throw new InvalidOperationException("Test set was not created!");
            }

            TestSet.Reset();
            var testWatch = new Stopwatch();
            testWatch.Start();
            var result = _tester.Test(TestableNetwork.Clone(), TestSet);
            testWatch.Stop();

            if (Options.ReportMesages)
            {
                Options.ProgressWriter?.Message($"=========\n\tTested in:\t{testWatch.Elapsed.TotalSeconds:0.0000}s\n{result.GetReport()}\n=========\n");
            }
        }

        private void TrainingSetOnDataSetReset(object sender, EventArgs eventArgs)
        {
            // An epoch was reached and data set rolled over.
            //OnMessage($"Epoch reached on iteration {Iteration}");
            Epoch++;
            Options.ProgressWriter?.ItemComplete();
            
            if (_tester != null && Options.RunTests?.ShouldDoOnEpoch(Epoch) == true)
            {
                RunTest();
            }

            // Check for epoch memory reset
            if (Options.ResetMemory?.ShouldDoOnEpoch(Epoch) == true)
            {
                ResetMemory();
            }

            OnEpochReached();

            _processedSamples = 0;
            Iteration = 0;
        }
    }
}