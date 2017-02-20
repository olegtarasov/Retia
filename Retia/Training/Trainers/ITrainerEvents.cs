using System;

namespace Retia.Training.Trainers
{
    public interface ITrainerEvents
    {
        event Action SequenceTrained;
        event Action EpochReached;
        long Epoch { get; }
        long Iteration { get; }
    }
}