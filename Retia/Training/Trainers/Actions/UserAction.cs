using System;

namespace Retia.Training.Trainers.Actions
{
    public class UserAction : PeriodicActionBase
    {
        private readonly Action _action;

        public UserAction(ActionSchedule schedule, Action action) : base(schedule)
        {
            _action = action;
        }

        protected override void DoAction()
        {
            _action();
        }
    }
}