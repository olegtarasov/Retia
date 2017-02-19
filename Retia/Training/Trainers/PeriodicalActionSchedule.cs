using Retia.Training.Trainers.Actions;

namespace Retia.Training.Trainers
{
    /// <summary>
    /// An action to scale learning rate.
    /// </summary>
    public class LearningRateScalingAction : PeriodicActionSchedule
    {
        public LearningRateScalingAction()
        {
        }

        public LearningRateScalingAction(int period, double scaleFactor)
        {
            EachEpochInternal(period);
            ScaleFactor = scaleFactor;
        }

        /// <summary>
        /// Learning rate scale factor.
        /// </summary>
        public double ScaleFactor { get; private set; }

        public void EachEpoch(int period, double scaleFactor)
        {
            EachEpochInternal(period);
            ScaleFactor = scaleFactor;
        }

        public void EachIteration(int period, double scaleFactor)
        {
            EachIterationInternal(period);
            ScaleFactor = scaleFactor;
        }
    }
}