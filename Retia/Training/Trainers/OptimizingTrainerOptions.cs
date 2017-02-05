namespace Retia.Training.Trainers
{
	public class OptimizingTrainerOptions : TrainerOptionsBase
	{
		public int ErrorFilterSize { get; set; } = 100;
	    public LearningRateScalingAction ScaleLearningRate { get; set; } = new LearningRateScalingAction(1, 0.8);
	    public int SequenceLength { get; set; } = 50;
	}
}