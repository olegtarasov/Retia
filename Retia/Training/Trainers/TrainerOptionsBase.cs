namespace Retia.Training.Trainers
{
    public abstract class TrainerOptionsBase
    {
        public int MaxEpoch { get; set; } = 80;
        public bool ReportMesages { get; set; } = false;

        public IterationAction ReportProgress { get; } = new IterationAction(10);
        public IterationAction RunUserTests { get; } = new IterationAction();
        public IterationAction RunTests { get; } = new IterationAction(100);
        public MultiPeriodAction ResetMemory { get; set; } = new MultiPeriodAction(1, PeriodType.Epoch);
    }
}