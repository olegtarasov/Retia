using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using OxyPlot;
using OxyPlot.Wpf;
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
        #region Dispatch

        protected void Dispatch(Action action)
        {
            Application.Current.Dispatcher.Invoke(action);
        }

        #endregion

        private readonly OptimizingTrainer<T> _trainer;

        private Task _trainingTask;
        private CancellationTokenSource _tokenSource;
        
        public TypedTrainingModel(OptimizingTrainer<T> trainer)
        {
            _trainer = trainer;

            OptionsModel = GetTrainOptionsModel();
            OptionsModel.StartResumeCommand = new RelayCommand(StartResume, CanStartResume);
            OptionsModel.PauseCommand = new RelayCommand(Pause, CanPause);
            OptionsModel.StopCommand = new RelayCommand(Stop, CanStop);
            OptionsModel.ApplyOptionsCommand = new RelayCommand(ApplyOptions);

            ReportModel = new TrainingReportModel();

            _trainer.TrainingStateChanged += TrainerOnTrainingStateChanged;
            _trainer.TrainReport += TrainerOnTrainReport;
        }

        public void ExportErrorPlot(Stream stream, int width, int height)
        {
            Dispatch(() => PngExporter.Export(ReportModel.PlotModel, stream, width, height, OxyColors.White));
        }

        private bool CanStop(object o)
        {
            return _trainer.IsTraining && !_trainer.IsPaused;
        }

        private void Stop(object o)
        {
            _trainer.Stop();
        }

        private void TrainerOnTrainingStateChanged(object sender, EventArgs eventArgs)
        {
            Dispatch(() =>
            {
                OptionsModel.StartResumeCommand.RaiseCanExecuteChanged();
                OptionsModel.PauseCommand.RaiseCanExecuteChanged();
                OptionsModel.StopCommand.RaiseCanExecuteChanged();
            });
        }

        private void ApplyOptions(object o)
        {
            // TODO: Apply options
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
            Dispatch(() =>
            {
                ReportModel.UpdateReport(e);
                OptionsModel.LearningRate = e.LearningRate;
            });
        }

        private TrainOptionsModel GetTrainOptionsModel()
        {
            return new TrainOptionsModel
                   {
                       ErrorFilterSize = _trainer.Options.ErrorFilterSize,
                       //LearningRate = _trainer.LearningRate,
                       //LearningRateScaleFactor = ((ProportionalLearningRateScaler)_trainer.Options.LearningRateScaler)?.ScalingFactor,
                       LearningRateScalePeriod = _trainer.Options.LearningRateScaler?.Schedule.Period,
                       MaxEpoch = _trainer.Options.MaxEpoch
                   };
        }
    }
}