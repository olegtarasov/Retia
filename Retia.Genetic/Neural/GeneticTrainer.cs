using System;
using System.Diagnostics;
using System.Linq;
using Retia.Genetic.Breeders;
using Retia.Genetic.Mutators;
using Retia.Genetic.Selectors;
using Retia.Neural;
using Retia.Training.Testers;
using Retia.Training.Trainers;

namespace Retia.Genetic.Neural
{
    public class GeneticTrainer : TrainerBase<GeneticTrainerOptions, GeneticReportEventArgs>
    {
        private readonly Evolver<EvolvableNet> _evolver;
        private EvolvableNet _alpha;

        private TimeSpan _wtc, _ctw;

        public GeneticTrainer(ITester tester, Evolver<EvolvableNet> evolver, GeneticTrainerOptions options) : base(null, tester, options)
        {
	        _evolver = evolver;
        }

        public override NeuralNet TestableNetwork => _alpha;

	    protected override GeneticReportEventArgs GetTrainingReport(bool userTest)
        {
            return new GeneticReportEventArgs(Epoch, Iteration, _evolver.MaxFitness);
        }

        protected override string GetTrainingReportMessage()
        {
            return $"\tMax fitness:\t{_evolver.MaxFitness}\n" +
                $"\tFit:\t{_evolver.PerformanceCounters[CounterType.Fitness].TotalSeconds:0.0000} s\n" +
                $"\tSort:\t{_evolver.PerformanceCounters[CounterType.Sort].TotalSeconds:0.0000} s\n" +
                $"\tWTC:\t{_wtc.TotalSeconds:0.0000} s\n" +
                $"\tBreed:\t{_evolver.PerformanceCounters[CounterType.Breed].TotalSeconds:0.0000} s\n" +
                $"\tMutate:\t{_evolver.PerformanceCounters[CounterType.Mutate].TotalSeconds:0.0000} s\n" +
                $"\tCTW:\t{_ctw.TotalSeconds:0.0000} s\n";
        }

        protected override void TrainIteration()
        {
            Epoch = Iteration;

            _evolver.CalculateFitness();
            _alpha = _evolver.Population[0];

            var watch = new Stopwatch();
            watch.Start();
            foreach (var net in _evolver.Population)
            {
                net.WeightsToChromosome();
            }
            watch.Stop();

            _wtc = watch.Elapsed;

            _evolver.Breed();
            _evolver.Mutate();

            watch.Restart();
            foreach (var net in _evolver.Population)
            {
                net.ChromosomeToWeights();
            }
            watch.Stop();

            _ctw = watch.Elapsed;
        }

        protected override void ResetMemory()
        {
            foreach (EvolvableNet net in _evolver.Population)
            {
                net.ResetMemory();
            }
        }
    }
}