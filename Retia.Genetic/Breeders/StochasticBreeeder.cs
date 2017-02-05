using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Retia.Genetic.Selectors;
using Retia.RandomGenerator;

namespace Retia.Genetic.Breeders
{
    public class StochasticBreeeder<T> : BreederBase<T> where T : IEvolvable
    {
        public StochasticBreeeder(SelectorBase<T> selector) : base(selector)
        {
        }

        public override List<T> Breed(List<T> source, int targetCount, double maxFitness)
        {
            var rnd = SafeRandom.Generator;
            var result = new ConcurrentQueue<T>();
            int chromosomeLen = source[0].Chromosome.Length;

            for (int idx = 0; idx < targetCount; idx++)
            {

                var selected = Selector.GetUniqueIndividuals(source, 2, maxFitness);

                var parent1 = selected[0];
                var parent2 = selected[1];

                var breed = (T)selected[0].Clone();
                //Console.WriteLine("-----------");
                for (int i = 0; i < chromosomeLen; i++)
                {
                    var gene1 = parent2.Chromosome[i];
                    var gene2 = parent1.Chromosome[i];

                    var prob = 0.5;//(i < chromosomeLen/2) ? 0.8 : 0.2;

                    if (rnd.NextDouble() < prob)
                        breed.Chromosome[i] = gene1;
                    /*
                    //visualisation
                    if (male.Chromosome[i] == gene1)
                        Console.Write((char) 0x2593);
                    else
                        Console.Write((char) 0x2591);*/
                }
                //Console.WriteLine("\n-----------");

                breed.Fitness = 0;
                result.Enqueue(breed);
            }

            return result.ToList();
        }
    }
}
