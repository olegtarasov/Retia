using System;
using System.Diagnostics;
using System.Linq;
using Retia.Genetic.Breeders;
using Retia.Genetic.Mutators;
using Retia.Genetic.Selectors;
using Retia.Neural;
using Retia.Training.Trainers;

namespace Retia.Genetic.Neural
{
    public class GeneticTrainer : TrainerBase<float, GeneticTrainerOptions, GeneticReportEventArgs, GeneticSession>
    {
        private readonly Evolver<EvolvableNet> _evolver;
        private EvolvableNet _alpha;

        private TimeSpan _wtc, _ctw;

        public GeneticTrainer(Evolver<EvolvableNet> evolver, GeneticTrainerOptions options) : base(options, new GeneticSession(null))
        {
	        _evolver = evolver;
        }

        public virtual NeuralNet<float> TestableNetwork => _alpha;

	    protected override GeneticReportEventArgs GetAndFlushTrainingReport()
        {
            return new GeneticReportEventArgs(Session, _evolver.MaxFitness);
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
            Session.Epoch = Session.Iteration;

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