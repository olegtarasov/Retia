using System;
using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers
{
    public interface ITrainerEvents
    {
        event Action<TrainingSessionBase> SequenceTrained;
        event Action<TrainingSessionBase> EpochReached;
    }
}