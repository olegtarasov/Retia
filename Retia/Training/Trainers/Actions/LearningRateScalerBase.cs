using Retia.Optimizers;

namespace Retia.Training.Trainers.Actions
{
    public abstract class LearningRateScalerBase : PeriodicActionBase
    {
        protected readonly IOptimizer Optimizer;

        protected float InitialRate;
        
        public LearningRateScalerBase(ActionSchedule schedule, IOptimizer optimizer) : base(schedule)
        {
            Optimizer = optimizer;
            InitialRate = optimizer.LearningRate;
        }

        public override void Reset()
        {
            base.Reset();
            InitialRate = Optimizer.LearningRate;
        }
    }
}