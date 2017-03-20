using System;
using Retia.Training.Trainers.Sessions;

namespace Retia.Training.Trainers
{
    public abstract class TrainReportEventArgsBase<T> : EventArgs where T : TrainingSessionBase
    {
        protected TrainReportEventArgsBase(T session)
        {
            Session = session;
        }

        public T Session { get; set; }
    }
}