using Retia.Training.Trainers;

namespace Retia.Genetic.Neural
{
    public class GeneticReportEventArgs : TrainReportEventArgsBase
    {
        public GeneticReportEventArgs(long epoch, long iteration, double maxFitness) : base(epoch, iteration)
        {
            MaxFitness = maxFitness;
        }

        public double MaxFitness { get; set; }
    }
}