using System.Windows.Controls.Primitives;
using System.Windows.Input;
using PropertyChanged;

namespace Retia.Gui.Models
{
    [ImplementPropertyChanged]
    public class TrainOptionsModel
    {
        public int ErrorFilterSize { get; set; } = 100;
        public float LearningRate { get; set; } = 0.0001f;
        public int LearningRateScalePeriod { get; set; } = 50;
        public float LearningRateScaleFactor { get; set; } = 0.01f;
        public int MaxEpoch { get; set; } = 100000;

        public RelayCommand StartResumeCommand { get; set; }
        public RelayCommand PauseCommand { get; set; }
        public RelayCommand ApplyOptionsCommand { get; set; }
    }
}