using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers.Actions
{
    public abstract class LearningRateScalerBase : TypedPeriodicActionBase<OptimizingSession>
    {
        public LearningRateScalerBase(ActionSchedule schedule) : base(schedule)
        {
        }
    }
}