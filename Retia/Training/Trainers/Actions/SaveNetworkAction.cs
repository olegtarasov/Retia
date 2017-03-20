using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers.Actions
{
    public class SaveNetworkAction : TypedPeriodicActionBase<OptimizingSession>
    {
        public SaveNetworkAction(ActionSchedule schedule, int versionsToKeep = 5) : base(schedule)
        {
            VersionsToKeep = versionsToKeep;
        }

        public int VersionsToKeep { get; set; }

        protected override void DoAction(OptimizingSession session)
        {
            session.SaveNetwork(VersionsToKeep);
        }
    }
}