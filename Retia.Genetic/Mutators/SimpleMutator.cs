using System.Collections.Generic;
using Retia.Genetic.Selectors;
using Retia.RandomGenerator;

namespace Retia.Genetic.Mutators
{
    /// <summary>
    /// A simple mutator which transforms the chromosomes.
    /// </summary>
    public class SimpleMutator<T> : MutatorBase<T> where T : IEvolvable
	{
        /// <summary>
        /// The scale to which to mutate the individual. Normally wuold be 0..1.
        /// </summary>
		public double MutationScale { get; set; }

        /// <summary>
        /// Population percent to mutate. 0..1.
        /// </summary>
		public double PopulationPercent { get; set; }

        /// <summary>
        /// The amount of bits in a chromosome to mutate. 0..1.
        /// </summary>
        public double ChromosomeMutationPercent { get; set; }

	    public SimpleMutator(SelectorBase<T> selector) : base(selector)
	    {
	    }

	    public override void Mutate(List<T> population, double maxFitness)
		{
		    var rnd = SafeRandom.Generator;
			int popCnt = (int)(population.Count * PopulationPercent);
			int chromeCnt = (int)(population[0].Chromosome.Length * ChromosomeMutationPercent);

	        foreach (var dude in Selector.GetUniqueIndividuals(population, popCnt, maxFitness))
	        {
                for (int i = 0; i < chromeCnt; i++)
                {
                    int chromeIdx = rnd.Next(dude.Chromosome.Length);
                    dude.Chromosome[chromeIdx] += dude.Chromosome[chromeIdx] * (float)rnd.NextDouble(-MutationScale, MutationScale);
                }
            }
		}
	}
}