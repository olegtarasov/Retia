using System;
using System.Collections.Generic;
using Retia.RandomGenerator;

namespace Retia.Genetic.Selectors
{
    /// <summary>
    /// Faster implementation of the roulette wheel, as per
    /// http://arxiv.org/pdf/1109.3627v2.pdf
    /// </summary>
    public abstract class StochasticAcceptanceSelectorBase<T> : SelectorBase<T> where T : class, IEvolvable
    {
        public override T GetIndividual(List<T> population, double maxFitness)
        {
            var rnd = SafeRandom.Generator;
            int popSize = population.Count;
            int sanityCheck = 0;

            while (true)
            {
                sanityCheck++;
                if (sanityCheck > popSize * 100)
                {
                    throw new InvalidOperationException($"Couldn't choose an individual in {popSize} attempts!");
                }

                int idx = rnd.Next(popSize);
                var individual = population[idx];

                if (ShouldAccept(rnd, individual.Fitness, maxFitness))
                {
                    return individual;
                }
            }
        }

        /// <summary>
        /// Tests whether we should accept selected individual.
        /// </summary>
        /// <param name="rnd">Random generator</param>
        /// <param name="fitness">Individual fitness.</param>
        /// <param name="maxFitness">Maximum population fitness.</param>
        /// <returns>True if the individual should be aacepted.</returns>
        protected abstract bool ShouldAccept(IRandom rnd, double fitness, double maxFitness);
    }

    public class StochasticAcceptanceSelector<T> : StochasticAcceptanceSelectorBase<T> where T : class, IEvolvable
    {
        protected override bool ShouldAccept(IRandom rnd, double fitness, double maxFitness)
        {
            return rnd.NextDouble() < fitness / maxFitness;
        }
    }

    /// <summary>
    /// This selector favors less fit individuals. You can use it to mutate less fit individuals.
    /// </summary>
    public class InverseStochasticAcceptanceSelector<T> : StochasticAcceptanceSelectorBase<T> where T : class, IEvolvable
    {
        protected override bool ShouldAccept(IRandom rnd, double fitness, double maxFitness)
        {
            return rnd.NextDouble() >= fitness / maxFitness;
        }
    }
}