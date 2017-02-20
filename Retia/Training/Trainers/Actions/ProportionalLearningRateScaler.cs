using Retia.Optimizers;

namespace Retia.Training.Trainers.Actions
{
    public class ProportionalLearningRateScaler : LearningRateScalerBase
    {
        private int _scalingTicks = 0;

        public ProportionalLearningRateScaler(ActionSchedule schedule, IOptimizer optimizer, float scalingFactor) : base(schedule, optimizer)
        {
            ScalingFactor = scalingFactor;
        }

        public float ScalingFactor { get; set; }

        public override void Reset()
        {
            base.Reset();
            _scalingTicks = 0;
        }

        protected override void DoAction()
        {
            _scalingTicks++;
            Optimizer.LearningRate = InitialRate / (1.0f + _scalingTicks * ScalingFactor);
        }
    }
}