using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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

        private List<double> _errors;
        private MAV _mav;
        private double _lastError;

        private double _dErr = 0;
        private int _processedSamples = 0;
        
        public OptimizingTrainer(NeuralNet<T> network, OptimizerBase<T> optimizer, IDataProvider<T> dataProvider, ITester<T> tester, OptimizingTrainerOptions options) : base(dataProvider, tester, options)
        {
            _network = network;
            _optimizer = optimizer;
            
            if (DataProvider.TrainingSet != null)
            {
                DataProvider.TrainingSet.DataSetReset += TrainingSetOnDataSetReset;
            }
        }

        public override NeuralNet<T> TestableNetwork => _network;

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

            result = DataProvider.TrainingSet.GetNextSamples(Options.SequenceLength);

            if (result == null)
            {
                // A new epoch has come, try to get the sequence again
                result = DataProvider.TrainingSet.GetNextSamples(Options.SequenceLength);

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
                filteredError = _mav.Filter(error);
            }
            else
            {
                filteredError = error;
            }

            _errors.Add(filteredError);
            _dErr = filteredError - _lastError;
            _lastError = filteredError;
        }

        protected override string GetIterationProgress()
        {
            if (DataProvider.TrainingSet.SampleCount > 0)
            {
                return $"I:{Iteration}/{DataProvider.TrainingSet.SampleCount / Options.SequenceLength}|{GetIterationProgressBar()}|";
            }

            return base.GetIterationProgress();
        }

        private string GetIterationProgressBar()
        {
            int n = (int)Math.Ceiling((_processedSamples / (double)DataProvider.TrainingSet.SampleCount) * 50);
            var sb = new StringBuilder(50);
            sb.Append('=', n).Append(' ', 50 - n);
            return sb.ToString();
        }

        protected override void DataProviderOnTrainingSetChanged(object sender, DataSetChangedArgs<T> e)
        {
            if (e.OldSet != null)
            {
                e.OldSet.DataSetReset -= TrainingSetOnDataSetReset;
            }

            if (e.NewSet != null)
            {
                e.NewSet.DataSetReset += TrainingSetOnDataSetReset;
            }
        }

        protected override void InitTraining()
        {
            base.InitTraining();
            DataProvider.CreateTrainingSet();
            DataProvider.CreateTestSet();
            _errors = new List<double>();
            _mav = Options.ErrorFilterSize > 0 ? new MAV(Options.ErrorFilterSize) : null;
            _lastError = 0;
            _processedSamples = 0;
            
            StatusWriter?.Message($"Sequence length: {Options.SequenceLength}");
            StatusWriter?.Message($"Using network with total param count {_network.TotalParamCount}");
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
        }

        protected override void ResetMemory()
        {
            StatusWriter?.Message($"Network memory reset on iteration {Iteration}, epoch {Epoch}");
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

        private void TrainingSetOnDataSetReset(object sender, EventArgs eventArgs)
        {
            // An epoch was reached and data set rolled over.
            //OnMessage($"Epoch reached on iteration {Iteration}");
            Epoch++;
            StatusWriter?.NewLine();
            OnEpochReached();

            _processedSamples = 0;
            Iteration = 0;

            // Check for epoch memory reset
            if (Options.ResetMemory.ShouldDoOnEpoch(Epoch))
            {
                ResetMemory();
            }
        }
    }
}