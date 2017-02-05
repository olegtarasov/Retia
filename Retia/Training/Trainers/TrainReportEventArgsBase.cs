using System;

namespace Retia.Training.Trainers
{
    public abstract class TrainReportEventArgsBase : EventArgs
    {
        protected TrainReportEventArgsBase(double epoch, long iteration)
        {
            Epoch = epoch;
            Iteration = iteration;
        }

        public double Epoch { get; set; }
        public long Iteration { get; set; }
    }
}