namespace Retia.Training.Trainers.Actions
{
    /// <summary>
    /// Frequency type.
    /// </summary>
    public enum PeriodType
    {
        /// <summary>
        /// The action is not performed.
        /// </summary>
        None,

        /// <summary>
        /// The frequency mesures in iterations.
        /// </summary>
        Iteration,

        /// <summary>
        /// The frequency measures in epochs.
        /// </summary>
        Epoch
    }
    
    /// <summary>
    /// A periodical action with a switch and a period.
    /// </summary>
    public class ActionSchedule
    {
        public ActionSchedule() : this(0, PeriodType.None)
        {
        }

        public ActionSchedule(int period, PeriodType periodType)
        {
            Period = period;
            PeriodType = periodType;
        }

        public static ActionSchedule Disabled => new ActionSchedule(0, PeriodType.None);

        /// <summary>
        /// Gets or sets whether action is enabled.
        /// </summary>
        public bool IsEnabled => PeriodType != PeriodType.None;

        /// <summary>
        /// The frequency at which the action is performed.
        /// </summary>
        public int Period { get; }

        /// <summary>
        /// Frequency type.
        /// </summary>
        public PeriodType PeriodType { get; }

        public bool ShouldDoOnEpoch(long epoch)
        {
            return PeriodType == PeriodType.Epoch && epoch % Period == 0;
        }

        public bool ShouldDoOnIteration(long iteration)
        {
            return PeriodType == PeriodType.Iteration && iteration % Period == 0;
        }
    }

    public class EachIteration : ActionSchedule
    {
        public EachIteration(int period) : base(period, PeriodType.Iteration)
        {
        }
    }

    public class EachEpoch : ActionSchedule
    {
        public EachEpoch(int period) : base(period, PeriodType.Epoch)
        {
        }
    }
}