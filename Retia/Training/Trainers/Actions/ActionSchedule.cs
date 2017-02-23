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

        /// <summary>
        /// Gets or sets whether action is enabled.
        /// </summary>
        public bool IsEnabled => PeriodType != PeriodType.None;

        /// <summary>
        /// The frequency at which the action is performed.
        /// </summary>
        public int Period { get; protected set; }

        /// <summary>
        /// Frequency type.
        /// </summary>
        public PeriodType PeriodType { get; protected set; }

        public void Never()
        {
            PeriodType = PeriodType.None;
        }

        public void EachIteration(int period)
        {
            PeriodType = PeriodType.Iteration;
            Period = period;
        }

        public void EachEpoch(int period)
        {
            PeriodType = PeriodType.Epoch;
            Period = period;
        }

        public bool ShouldDoOnEpoch(long epoch)
        {
            return PeriodType == PeriodType.Epoch && epoch % Period == 0;
        }

        public bool ShouldDoOnIteration(long iteration)
        {
            return PeriodType == PeriodType.Iteration && iteration % Period == 0;
        }
    }
}