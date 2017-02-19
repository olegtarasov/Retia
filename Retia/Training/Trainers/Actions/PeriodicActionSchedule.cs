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
    public abstract class PeriodicActionSchedule
    {
        /// <summary>
        /// Gets or sets whether action is enabled.
        /// </summary>
        public bool IsEnabled { get { return PeriodType == PeriodType.None; } }

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

        protected void EachIterationInternal(int period)
        {
            PeriodType = PeriodType.Iteration;
            Period = period;
        }

        protected void EachEpochInternal(int period)
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

    public class IterationActionSchedule : PeriodicActionSchedule
    {
        public IterationActionSchedule()
        {
        }

        public IterationActionSchedule(int period)
        {
            EachIteration(period);
        }

        public void EachIteration(int period)
        {
            EachIterationInternal(period);
        }
    }

    public class EpochActionSchedule : PeriodicActionSchedule
    {
        public EpochActionSchedule()
        {
        }

        public EpochActionSchedule(int period)
        {
            EachEpoch(period);
        }

        public void EachEpoch(int period)
        {
            EachEpochInternal(period);
        }
    }

    public class MultiPeriodActionSchedule : IterationActionSchedule
    {
        public MultiPeriodActionSchedule()
        {
        }

        public MultiPeriodActionSchedule(int period, PeriodType type) : base(period)
        {
            PeriodType = type;
        }

        public void EachEpoch(int period)
        {
            EachEpochInternal(period);
        }
    }
}