using System;
using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers
{
	public interface ITrainingReporter<TReport, TSession> 
        where TReport : TrainReportEventArgsBase<TSession>
        where TSession : TrainingSessionBase
    {
		event EventHandler<TReport> TrainReport;
	}
}