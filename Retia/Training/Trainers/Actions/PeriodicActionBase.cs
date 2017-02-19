namespace Retia.Training.Trainers.Actions
{
    public abstract class PeriodicActionBase
    {
        protected PeriodicActionBase(PeriodicActionSchedule schedule)
        {
            Schedule = schedule;
        }

        public PeriodicActionSchedule Schedule { get; private set; }
    }
}