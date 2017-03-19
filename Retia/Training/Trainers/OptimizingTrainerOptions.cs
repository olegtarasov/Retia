using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
	public class OptimizingTrainerOptions : TrainerOptionsBase
	{
	    public OptimizingTrainerOptions(int sequenceLength)
	    {
	        SequenceLength = sequenceLength;
	    }

	    public int ErrorFilterSize { get; set; } = 100;
	    public int SequenceLength { get; }
	    public LearningRateScalerBase LearningRateScaler { get; set; }
	}
}