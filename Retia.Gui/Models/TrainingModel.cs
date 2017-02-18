using System;
using PropertyChanged;
using Retia.Integration;
using Retia.Training.Trainers;

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

        public TypedTrainingModel(OptimizingTrainer<T> trainer)
        {
            _trainer = trainer;

            OptionsModel = GetTrainOptionsModel();
            ReportModel = new TrainingReportModel();

            _trainer.TrainReport += TrainerOnTrainReport;
            _trainer.Message += TrainerOnMessage;
        }

        private void TrainerOnMessage(object sender, LogEventArgs e)
        {
            ReportModel.AddMessage(e.Message);
        }

        private void TrainerOnTrainReport(object sender, OptimizationReportEventArgs e)
        {
            ReportModel.UpdateReport(e);
        }

        private TrainOptionsModel GetTrainOptionsModel()
        {
            return new TrainOptionsModel
                   {
                       ErrorFilterSize = _trainer.Options.ErrorFilterSize,
                       LearningRate = _trainer.LearningRate,
                       LearningRateScaleFactor = _trainer.Options.ScaleLearningRate.ScaleFactor,
                       LearningRateScalePeriod = _trainer.Options.ScaleLearningRate.Period,
                       MaxEpoch = _trainer.Options.MaxEpoch
                   };
        }
    }
}