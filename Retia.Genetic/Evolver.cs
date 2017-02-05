using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Retia.Genetic.Breeders;
using Retia.Genetic.Generators;
using Retia.Genetic.Mutators;

namespace Retia.Genetic
{
    public enum CounterType
    {
        Fitness,
        Sort,
        Breed,
        Mutate
    }

	public class Evolver<T> where T : class, IEvolvable
	{
	    private readonly BreederBase<T> _breeder;
	    private readonly MutatorBase<T> _mutator;
	    private readonly Func<T, double> _fitnessFunc;
	    private readonly int _populationSize;
	    private readonly Dictionary<CounterType, TimeSpan> _performanceCounters = new Dictionary<CounterType, TimeSpan>();

	    public Evolver(	IPopulationGenerator<T> generator,
						int populationSize,
						BreederBase<T> breeder,
						MutatorBase<T> mutator,
						Func<T, double> fitnessFunc) : this(GeneratePopulation(generator, populationSize), breeder, mutator, fitnessFunc)
		{
        }

	    public Evolver(List<T> initialPopulation,
	                   BreederBase<T> breeder,
	                   MutatorBase<T> mutator,
	                   Func<T, double> fitnessFunc)
	    {
            _breeder = breeder;
	        _populationSize = initialPopulation.Count;
	        _mutator = mutator;
	        _fitnessFunc = fitnessFunc;

	        Population = initialPopulation;

	        IsParallel = true;

	        //default value, % of old population which won't make it to new one
	        DeathRate = 0.7;
	    }

	    public IReadOnlyDictionary<CounterType, TimeSpan> PerformanceCounters => _performanceCounters;

	    public double MaxFitness { get; private set; }

	    public bool IsParallel { get; set; }

	    public double DeathRate { get; set; }

	    public List<T> Population { get; protected set; }

	    public void CalculateFitness()
		{
            //you should definitely reset MaxFitnes, as population with this value no longer exist
		    MaxFitness = 0;

            var watch = new Stopwatch();
            watch.Start();
            
			CalculatePopulationFitness();

            watch.Stop();

            _performanceCounters[CounterType.Fitness] = watch.Elapsed;

            watch.Restart();
            Population.Sort();
            watch.Stop();

            _performanceCounters[CounterType.Sort] = watch.Elapsed;

            MaxFitness = Population[0].Fitness;
		}

	    public void Breed()
		{
            var watch = new Stopwatch();
            watch.Start();
            
            //Should be already sorted and evaluated population at this point!
            var newPopulation = new List<T>(Population.Count);
	        var killCnt = (int)(DeathRate* Population.Count);
            for(int i=0;i<Population.Count-killCnt;i++)
                newPopulation.Add((T)Population[i]);
            //newPopulation.RemoveRange(newPopulation.Count-killCnt, killCnt);

            var breeded = _breeder.Breed(Population, _populationSize-newPopulation.Count, MaxFitness);
            newPopulation.AddRange(breeded);
	        Population = newPopulation;

            watch.Stop();

		    _performanceCounters[CounterType.Breed] = watch.Elapsed;
		}

	    public void Mutate()
		{
            var watch = new Stopwatch();
            watch.Start();

            _mutator.Mutate(Population, MaxFitness);

            watch.Stop();

		    _performanceCounters[CounterType.Mutate] = watch.Elapsed;
		}

	    protected virtual void CalculatePopulationFitness()
		{
			if (IsParallel)
			{
				Parallel.ForEach(Population, (dude, state) =>
				{
					dude.Fitness = _fitnessFunc(dude);
				});
			}
			else
			{
				foreach (var dude in Population)
				{
					dude.Fitness = _fitnessFunc(dude);
				}
			}
		}

	    private static List<T> GeneratePopulation(IPopulationGenerator<T> generator, int populationSize)
	    {
            var result = new List<T>();
            // Generate intial population
            for (int i = 0; i < populationSize; i++)
                result.Add(generator.GenerateIndividual());

	        return result;
	    }
	}
}