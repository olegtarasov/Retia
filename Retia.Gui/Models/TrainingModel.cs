using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using PropertyChanged;
using Retia.Integration;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Actions;

namespace Retia.Gui.Models
{
    [ImplementPropertyChanged]
    public abstract class TrainingModelBase
    {
        public TrainOptionsModel OptionsModel { get; set; }
        public TrainingReportModel ReportModel { get; set; }
    }

    [ImplementPropertyChanged]
    public class TypedTrainingModel<T> : TrainingModelBase where T : struct, IEquatable<T>, IFormattable
    {
        private readonly OptimizingTrainer<T> _trainer;

        private Task _trainingTask;
        private CancellationTokenSource _tokenSource;
        
        public TypedTrainingModel(OptimizingTrainer<T> trainer)
        {
            _trainer = trainer;

            OptionsModel = GetTrainOptionsModel();
            OptionsModel.StartResumeCommand = new RelayCommand(StartResume, CanStartResume);
            OptionsModel.PauseCommand = new RelayCommand(Pause, CanPause);
            OptionsModel.ApplyOptionsCommand = new RelayCommand(ApplyOptions);

            ReportModel = new TrainingReportModel();

            _trainer.TrainReport += TrainerOnTrainReport;
        }

        private void ApplyOptions(object o)
        {
            _trainer.Options.ErrorFilterSize = OptionsModel.ErrorFilterSize;
            if (Math.Abs(_trainer.LearningRate - OptionsModel.LearningRate) > 1e-5)
            {
                _trainer.LearningRate = OptionsModel.LearningRate;
            }
            _trainer.Options.LearningRateScaler.Schedule = new EachIteration(OptionsModel.LearningRateScalePeriod.GetValueOrDefault());
            ((ProportionalLearningRateScaler)_trainer.Options.LearningRateScaler).ScalingFactor = OptionsModel.LearningRateScaleFactor.GetValueOrDefault();
            _trainer.Options.MaxEpoch = OptionsModel.MaxEpoch;
        }

        private bool CanPause(object o)
        {
            return _trainer.IsTraining && !_trainer.IsPaused;
        }

        private bool CanStartResume(object o)
        {
            return !_trainer.IsTraining || _trainer.IsPaused;
        }

        private void Pause(object o)
        {
            _trainer.Pause();
            OptionsModel.PauseCommand.RaiseCanExecuteChanged();
            OptionsModel.StartResumeCommand.RaiseCanExecuteChanged();
        }

        private void StartResume(object o)
        {
            if (_trainer.IsTraining && _trainer.IsPaused)
            {
                _trainer.Resume();
            }
            else if (!_trainer.IsTraining)
            {
                _tokenSource = new CancellationTokenSource();
                _trainingTask = _trainer.Train(_tokenSource.Token);
            }

            OptionsModel.PauseCommand.RaiseCanExecuteChanged();
            OptionsModel.StartResumeCommand.RaiseCanExecuteChanged();
        }

        private void TrainerOnTrainReport(object sender, OptimizationReportEventArgs e)
        {
            ReportModel.UpdateReport(e);
            OptionsModel.LearningRate = e.LearningRate;
        }

        private TrainOptionsModel GetTrainOptionsModel()
        {
            return new TrainOptionsModel
                   {
                       ErrorFilterSize = _trainer.Options.ErrorFilterSize,
                       LearningRate = _trainer.LearningRate,
                       LearningRateScaleFactor = ((ProportionalLearningRateScaler)_trainer.Options.LearningRateScaler)?.ScalingFactor,
                       LearningRateScalePeriod = _trainer.Options.LearningRateScaler?.Schedule.Period,
                       MaxEpoch = _trainer.Options.MaxEpoch
                   };
        }
    }
}