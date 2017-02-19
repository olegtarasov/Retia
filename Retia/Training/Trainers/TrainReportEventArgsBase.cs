using System;

namespace Retia.Training.Trainers
{
    public abstract class TrainReportEventArgsBase : EventArgs
    {
        protected TrainReportEventArgsBase(long epoch, long iteration)
        {
            Epoch = epoch;
            Iteration = iteration;
        }

        public long Epoch { get; set; }
        public long Iteration { get; set; }
    }
}