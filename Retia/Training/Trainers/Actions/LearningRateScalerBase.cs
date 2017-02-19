namespace Retia.Training.Trainers.Actions
{
    public abstract class LearningRateScalerBase : PeriodicActionBase
    {
        protected float InitialRate;
        
        public LearningRateScalerBase(MultiPeriodActionSchedule schedule) : base(schedule)
        {
        }

        public virtual void Initialize(float initialRate)
        {
            InitialRate = initialRate;
        }

        public abstract float ScaleLearningRate();
    }
}