using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Retia.Genetic.Selectors;
using Retia.RandomGenerator;

namespace Retia.Genetic.Breeders
{
    /// <summary>
    /// Breeds individuals by selecting two of them, then swapping
    /// random part of their chromosomes with each other. 
    /// For every individual pair the swapping point is the same.
    /// Swapping lenght is the same for the whole breeder and is
    /// defined by <see cref="SwapCrhomosomePercent"/> percentage of the chromosome.
    /// Selected individuals are guaranteed not to be the same one individual. 
    /// 
    /// As per https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_crossover.
    /// </summary>
    public class TwoPointBreeder<T> : BreederBase<T> where T : IEvolvable
	{
		public double SwapCrhomosomePercent { get; set; }

		public TwoPointBreeder(SelectorBase<T> selector) : base(selector)
		{
		}

		public override List<T> Breed(List<T> source, int targetCount, double maxFitness)
		{
		    var rnd = SafeRandom.Generator;
			var result = new ConcurrentQueue<T>();
		    int chromosomeLen = source[0].Chromosome.Length;
		    int len = (int)Math.Ceiling(SwapCrhomosomePercent * chromosomeLen);
            int resCnt = targetCount / 2;

            for (int idx = 0; idx < resCnt; idx++)
            {
                int start = rnd.Next(chromosomeLen - len);
                var selected = Selector.GetUniqueIndividuals(source, 2, maxFitness);
                var tmp = new double[len];

				var male = (T)selected[0].Clone();
				var female = (T)selected[1].Clone();

                Array.Copy(male.Chromosome, start, tmp, 0, len);
                Array.Copy(female.Chromosome, start, male.Chromosome, start, len);
                Array.Copy(tmp, 0, female.Chromosome, start, len);

				result.Enqueue(male);
				result.Enqueue(female);
			}

			return result.ToList();
		}
	}
}