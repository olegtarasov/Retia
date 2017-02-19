using System.Collections.Generic;

namespace Retia.Training.Trainers
{
	public class OptimizationReportEventArgs : TrainReportEventArgsBase
	{
		public OptimizationReportEventArgs(List<double> errors, long epoch, long iteration, float learningRate) : base(epoch, iteration)
		{
			Errors = errors;
			LearningRate = learningRate;
		}

		public List<double> Errors { get; set; }
        public float LearningRate { get; set; }
	}
}