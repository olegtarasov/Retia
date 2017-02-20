using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
	public class OptimizingTrainerOptions : TrainerOptionsBase
	{
		public int ErrorFilterSize { get; set; } = 100;
	    public int SequenceLength { get; set; } = 50;
	    public LearningRateScalerBase LearningRateScaler { get; set; }
	}
}