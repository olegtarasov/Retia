using System.Collections.Generic;

namespace Retia.Training.Trainers
{
	public class OptimizationReportEventArgs : TrainReportEventArgsBase
	{
		public OptimizationReportEventArgs(List<double> errors, double epoch, long iteration, double learningRate) : base(epoch, iteration)
		{
			Errors = errors;
			LearningRate = learningRate;
		}

		public List<double> Errors { get; set; }
        public double LearningRate { get; set; }
	}
}