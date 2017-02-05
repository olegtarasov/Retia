using System;
using System.Collections.Generic;
using Retia.Genetic.Selectors;

namespace Retia.Genetic.Breeders
{
    /// <summary>
    /// Base class for a breeder.
    /// </summary>
    /// <typeparam name="T"></typeparam>
	public abstract class BreederBase<T> where T : IEvolvable
	{
        /// <summary>
        /// Individual selector.
        /// </summary>
		protected readonly SelectorBase<T> Selector;

		protected BreederBase(SelectorBase<T> selector)
		{
			Selector = selector;
		}

        /// <summary>
        /// Breed the population.
        /// </summary>
        /// <param name="source">Current population.</param>
        /// <param name="targetCount">Target population count.</param>
        /// <param name="maxFitness">Maximum population fitness.</param>
        /// <returns>New population.</returns>
        public abstract List<T> Breed(List<T> source, int targetCount, double maxFitness);
	}
}