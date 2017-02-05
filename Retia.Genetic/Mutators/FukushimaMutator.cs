using System.Collections.Generic;
using System.Linq;
using Retia.Genetic.Selectors;
using Retia.RandomGenerator;

namespace Retia.Genetic.Mutators
{
    /// <summary>
    /// The amount of mutation this class gives is proportional to
    /// subject's fitness. The less fit you are, the more you mutate.
    /// Individual selection is controlled by the selector, as always.
    /// </summary>
    public class FukushimaMutator<T> : MutatorBase<T> where T : IEvolvable
    {
        public double MinimumMutation { get; set; }
        public double MaximumMutation { get; set; }
        public double MinimumSpread { get; set; }
        public double MaximumSpread { get; set; }
        public double PopulationPercent { get; set; }

        public FukushimaMutator(SelectorBase<T> selector) : base(selector)
        {
        }

        public override void Mutate(List<T> population, double maxFitness)
        {
            var rnd = SafeRandom.Generator;

            foreach (var dude in Selector.GetUniqueIndividuals(population, (int)(population.Count * PopulationPercent), maxFitness))
            {
                double fitnessCoeff = 1 - dude.Fitness / maxFitness;
                int chromeCount = (int)((MinimumMutation + ((MaximumMutation - MinimumMutation) * fitnessCoeff)) * dude.Chromosome.Length);
                double spread = MinimumSpread + ((MaximumSpread - MinimumSpread) * fitnessCoeff);

                for (int i = 0; i < chromeCount; i++)
                {
                    int chromeIdx = rnd.Next(dude.Chromosome.Length);
                    dude.Chromosome[chromeIdx] += dude.Chromosome[chromeIdx] * (float)rnd.NextDouble(-spread, spread);
                }
            }
        }
    }
}