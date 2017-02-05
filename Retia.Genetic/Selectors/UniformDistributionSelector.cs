using System.Collections.Generic;
using Retia.RandomGenerator;

namespace Retia.Genetic.Selectors
{
    /// <summary>
    /// The simplest selector returning unbiased random individual.
    /// </summary>
    public class UniformDistributionSelector<T> : SelectorBase<T> where T : class, IEvolvable
    {
        public override T GetIndividual(List<T> population, double maxFitness)
        {
            return population[SafeRandom.Generator.Next(population.Count)];
        }
    }
}