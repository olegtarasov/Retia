using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
    public abstract class TrainerOptionsBase
    {
        public int MaxEpoch { get; set; } = 80;
        public bool ReportMesages { get; set; } = false;

        public ActionSchedule ReportProgress { get; } = new ActionSchedule(10, PeriodType.Iteration);
        public ActionSchedule RunTests { get; } = new ActionSchedule(100, PeriodType.Iteration);
        public ActionSchedule ResetMemory { get; set; } = new ActionSchedule(1, PeriodType.Epoch);
    }
}