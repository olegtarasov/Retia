using System.Collections.Generic;
using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers
{
	public class OptimizationReportEventArgs : TrainReportEventArgsBase<OptimizingSession>
	{
		public OptimizationReportEventArgs(OptimizingSession session, List<OptimizationError> errors, float learningRate) : base(session)
		{
			Errors = errors;
			LearningRate = learningRate;
		}

		public List<OptimizationError> Errors { get; set; }
        public float LearningRate { get; set; }
	}
}