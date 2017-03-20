using Retia.Optimizers;
using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers.Actions
{
    public class ProportionalLearningRateScaler : LearningRateScalerBase
    {
        private float _initialRate = float.NaN;
        private int _scalingTicks = 0;


        public ProportionalLearningRateScaler(ActionSchedule schedule, float scalingFactor) : base(schedule)
        {
            ScalingFactor = scalingFactor;
        }

        public float ScalingFactor { get; set; }

        public override void Reset()
        {
            base.Reset();
            _scalingTicks = 0;
            _initialRate = float.NaN;
        }

        protected override void DoAction(OptimizingSession session)
        {
            if (float.IsNaN(_initialRate))
            {
                _initialRate = session.Optimizer.LearningRate;
            }

            _scalingTicks++;
            session.Optimizer.LearningRate = _initialRate / (1.0f + _scalingTicks * ScalingFactor);
        }
    }
}