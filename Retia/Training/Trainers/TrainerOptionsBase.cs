using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
    public abstract class TrainerOptionsBase
    {
        public int MaxEpoch { get; set; } = 80;
        public bool ReportMesages { get; set; } = false;

        public IterationActionSchedule ReportProgress { get; } = new IterationActionSchedule(10);
        public IterationActionSchedule RunUserTests { get; } = new IterationActionSchedule();
        public IterationActionSchedule RunTests { get; } = new IterationActionSchedule(100);
        public MultiPeriodActionSchedule ResetMemory { get; set; } = new MultiPeriodActionSchedule(1, PeriodType.Epoch);
    }
}