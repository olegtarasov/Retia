using System;
using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers.Actions
{
    public class UserAction : PeriodicActionBase
    {
        private readonly Action _action;

        public UserAction(ActionSchedule schedule, Action action) : base(schedule)
        {
            _action = action;
        }

        protected override void DoAction(TrainingSessionBase session)
        {
            _action();
        }
    }
}