using Retia.Training.Trainers;

namespace Retia.Genetic.Neural
{
    public class GeneticReportEventArgs : TrainReportEventArgsBase<GeneticSession>
    {
        public GeneticReportEventArgs(GeneticSession session, double maxFitness) : base(session)
        {
            MaxFitness = maxFitness;
        }

        public double MaxFitness { get; set; }
    }
}