using Retia.Integration;
using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
    public abstract class TrainerOptionsBase
    {
        public int MaxEpoch { get; set; } = 80;
        public bool ReportMesages { get; set; } = false;
        public IProgressWriter ProgressWriter { get; set; }

        public ActionSchedule ReportProgress { get; set; } = new EachIteration(10);
        public ActionSchedule RunTests { get; set; } = ActionSchedule.Disabled;
        public ActionSchedule ResetMemory { get; set; } = new EachEpoch(1);
    }
}