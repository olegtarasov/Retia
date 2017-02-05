using System;

namespace Retia.Training.Trainers
{
	public interface ITrainingReporter<T> where T : TrainReportEventArgsBase
    {
		event EventHandler<T> TrainReport;
	}
}