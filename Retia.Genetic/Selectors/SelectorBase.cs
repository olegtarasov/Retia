using System;
using System.Collections.Generic;
using System.Linq;

namespace Retia.Genetic.Selectors
{
    /// <summary>
    /// The base class for individual selector.
    /// </summary>
    public abstract class SelectorBase<T> where T : IEvolvable
    {
        /// <summary>
        /// Gets a sigle individual according to selector rules.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="maxFitness">Maximum population fitness.</param>
        /// <returns>Individual.</returns>
        public abstract T GetIndividual(List<T> population, double maxFitness);

        /// <summary>
        /// Gets unique individuals according to selector rules.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="count">Number of individuals to select.</param>
        /// <param name="maxFitness">Maximum population fitness.</param>
        /// <returns>The list of individuals.</returns>
        public T[] GetUniqueIndividuals(List<T> population, int count, double maxFitness)
        {
            if (count > population.Count)
            {
                throw new ArgumentOutOfRangeException("count", "Can't select more individuals than population currently has.");
            }

            if (count == population.Count)
            {
                return population.ToArray();
            }

            var result = new HashSet<T>();

            while (result.Count < count)
            {
                var selected = GetIndividual(population, maxFitness);
                if (!result.Contains(selected))
                {
                    result.Add(selected);
                }
            }

            return result.ToArray();
        }
    }
}