namespace Retia.Training.Trainers.Actions
{
    public class ProportionalLearningRateScaler : LearningRateScalerBase
    {
        private int _scalingTicks = 0;

        public ProportionalLearningRateScaler(MultiPeriodActionSchedule schedule, float scalingFactor) : base(schedule)
        {
            ScalingFactor = scalingFactor;
        }

        public float ScalingFactor { get; set; }

        public override void Initialize(float initialRate)
        {
            base.Initialize(initialRate);
            _scalingTicks = 0;
        }

        public override float ScaleLearningRate()
        {
            return InitialRate / (1.0f + _scalingTicks * ScalingFactor);
        }

        public static ProportionalLearningRateScaler EachIteration(int period, float scalingFactor)
        {
            return new ProportionalLearningRateScaler(new MultiPeriodActionSchedule(period, PeriodType.Iteration), scalingFactor);
        }

        public static ProportionalLearningRateScaler EachEpoch(int period, float scalingFactor)
        {
            return new ProportionalLearningRateScaler(new MultiPeriodActionSchedule(period, PeriodType.Epoch), scalingFactor);
        }
    }
}