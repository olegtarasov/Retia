using Retia.Integration;
using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
    public abstract class TrainerOptionsBase
    {
        public int MaxEpoch { get; set; } = 80;
        public bool ReportMesages { get; set; } = false;
        public IProgressWriter ProgressWriter { get; set; }

        public ActionSchedule ReportProgress { get; } = new ActionSchedule(10, PeriodType.Iteration);
        public ActionSchedule RunTests { get; } = new ActionSchedule(0, PeriodType.None);
        public ActionSchedule ResetMemory { get; set; } = new ActionSchedule(1, PeriodType.Epoch);
    }
}