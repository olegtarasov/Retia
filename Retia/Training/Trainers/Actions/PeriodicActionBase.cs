using System;

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

        protected abstract void DoAction();

        private void TrainerEventsOnSequenceTrained()
        {
            if (Schedule.ShouldDoOnIteration(_trainerEvents.Iteration))
            {
                DoAction();
            }
        }

        private void TrainerEventsOnEpochReached()
        {
            if (Schedule.ShouldDoOnEpoch(_trainerEvents.Epoch))
            {
                DoAction();
            }
        }
    }
}