using System;
using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers.Actions
{
    public abstract class PeriodicActionBase
    {
        private ITrainerEvents _trainerEvents;

        protected PeriodicActionBase(ActionSchedule schedule)
        {
            Schedule = schedule;
        }

        public ActionSchedule Schedule { get; set; }

        public virtual void Reset()
        {
        }

        internal void Subscribe(ITrainerEvents trainerEvents)
        {
            _trainerEvents = trainerEvents;
            _trainerEvents.EpochReached += TrainerEventsOnEpochReached;
            _trainerEvents.SequenceTrained += TrainerEventsOnSequenceTrained;
        }

        internal void Unsubscribe()
        {
            _trainerEvents.EpochReached -= TrainerEventsOnEpochReached;
            _trainerEvents.SequenceTrained -= TrainerEventsOnSequenceTrained;
        }

        protected abstract void DoAction(TrainingSessionBase session);

        private void TrainerEventsOnSequenceTrained(TrainingSessionBase session)
        {
            if (Schedule.ShouldDoOnIteration(session.Iteration))
            {
                DoAction(session);
            }
        }

        private void TrainerEventsOnEpochReached(TrainingSessionBase session)
        {
            if (Schedule.ShouldDoOnEpoch(session.Epoch))
            {
                DoAction(session);
            }
        }
    }

    public abstract class TypedPeriodicActionBase<T> : PeriodicActionBase where T : TrainingSessionBase
    {
        public TypedPeriodicActionBase(ActionSchedule schedule) : base(schedule)
        {
        }

        protected override void DoAction(TrainingSessionBase session)
        {
            DoAction((T)session);
        }

        protected abstract void DoAction(T session);
    }
}