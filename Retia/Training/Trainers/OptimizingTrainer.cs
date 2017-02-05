﻿using System;
using System.Collections.Generic;
using System.Linq;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Optimizers;
using Retia.Training.Data;
using Retia.Training.Testers;

namespace Retia.Training.Trainers
{
    public class OptimizingTrainer: TrainerBase<OptimizingTrainerOptions, OptimizationReportEventArgs> 
    {
        protected readonly NeuralNet _network;
        private readonly OptimizerBase _optimizer;
        private readonly double _initialLr;

        private List<double> _errors;
        private Fir _fir;
        private double _lastError;

        private double _dErr = 0;

        public OptimizingTrainer(NeuralNet network, OptimizerBase optimizer, IDataProvider dataProvider, ITester tester, OptimizingTrainerOptions options) : base(dataProvider, tester, options)
        {
            _network = network;
            _optimizer = optimizer;
            _initialLr = _optimizer.LearningRate;

            if (DataProvider.TrainingSet != null)
            {
                DataProvider.TrainingSet.DataSetReset += TrainingSetOnDataSetReset;
            }
        }

        public override NeuralNet TestableNetwork => _network;

        protected virtual TrainingSequence GetTrainSamples()
        {
            // We can get TrainingSetOnDataSetReset during this call
            TrainingSequence result;

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
                filteredError = _fir.FilterSample(error);
                if (!_fir.FirstTurnComplete)
                {
                    filteredError = error;
                }
            }
            else
            {
                filteredError = error;
            }

            _errors.Add(filteredError);
            _dErr = filteredError - _lastError;
            _lastError = filteredError;
        }

        protected override void DataProviderOnTrainingSetChanged(object sender, DataSetChangedArgs e)
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
            _fir = Options.ErrorFilterSize > 0 ? new Fir(Options.ErrorFilterSize) : null;
            _lastError = 0;

            OnMessage($"Sequence length: {Options.SequenceLength}");
            OnMessage($"Using network with total param count {_network.TotalParamCount}");
        }

        protected override OptimizationReportEventArgs GetTrainingReport(bool userTest)
        {
            var result = new OptimizationReportEventArgs(_errors.ToList(), Epoch, Iteration, _optimizer.LearningRate);

            if (!userTest)
            {
                _errors.Clear();
            }

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
            _network.Optimize();
            ProcessError(error);
        }

        protected override void ResetMemory()
        {
            OnMessage($"Network memory reset on iteration {Iteration}, epoch {Epoch}");
            _network.ResetMemory();
        }

        protected override string GetTrainingReportMessage()
        {
            return $"\tLast error:\t{_lastError:0.0000}\n\tDelta error:\t{_dErr:0.0000}\n\tLearning rate:\t{_optimizer.LearningRate}";
        }

        private void TrainingSetOnDataSetReset(object sender, EventArgs eventArgs)
        {
            // An epoch was reached and data set rolled over.
            OnMessage($"Epoch reached on iteration {Iteration}");
            Epoch++;
            OnEpochReached();

            // Check for epoch memory reset
            if (Options.ResetMemory.ShouldDoOnEpoch(Epoch))
            {
                ResetMemory();
            }

            // Check for learning rate scaling
            if (Options.ScaleLearningRate.ShouldDoOnEpoch(Epoch))
            {
                _optimizer.LearningRate = (float)(_initialLr / (1.0 + Iteration * Options.ScaleLearningRate.ScaleFactor));
            }
        }
    }
}