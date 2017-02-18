using System.Windows.Controls.Primitives;
using PropertyChanged;
using Retia.Training.Trainers;

namespace Retia.Gui.Models
{
    [ImplementPropertyChanged]
    public class TrainOptionsModel
    {
        public int ErrorFilterSize { get; set; } = 100;
        public float LearningRate { get; set; } = 0.0001f;
        public int LearningRateScalePeriod { get; set; } = 50;
        public double LearningRateScaleFactor { get; set; } = 0.01f;
        public int MaxEpoch { get; set; } = 100000;
        //public int SequenceLenght { get; set; } = 2000;

        internal OptimizingTrainerOptions GetOptions()
        {
            var options = new OptimizingTrainerOptions
                          {
                              ErrorFilterSize = ErrorFilterSize,
                              ScaleLearningRate = new LearningRateScalingAction(LearningRateScalePeriod, LearningRateScaleFactor),
                              MaxEpoch = MaxEpoch,
                              ReportMesages = true,
                              //SequenceLength = SequenceLenght
                          };

            options.ReportProgress.EachIteration(5);
            options.ResetMemory.Never();
            options.RunTests.Never();
            options.RunUserTests.EachIteration(5);

            return options;
        }
    }
}