//using System.Collections.Generic;
//using System.Linq;
//using Retia.Genetic;
//using Xunit;
//using XunitShould;

//namespace Retia.Tests.Genetic
//{
//    public abstract class GeneticTestsBase
//    {
//        protected List<Evolvable> GetPopulation()
//        {
//            var result = new List<Evolvable>();

//            for (int i = 0; i < 10; i++)
//            {
//                result.Add(new Evolvable
//                           {
//                               Chromosome = Enumerable.Range(i, 10).Select(x => (float)x).ToArray()
//                           });
//            }

//            return result;
//        }

//        protected void AssertSourceChromosome(Evolvable dude, int startIdx)
//        {
//            dude.ShouldNotBeNull();
//            dude.Chromosome.ShouldNotBeNull();
//            dude.Chromosome.Length.ShouldEqual(10);
//            dude.Chromosome.ShouldEnumerateEqual(Enumerable.Range(startIdx, 10).Select(x => (float)x));
//        }

//        protected class Evolvable : IEvolvable
//        {
//            public double Fitness { get; set; }

//            public float[] Chromosome { get; set; }

//            public IEvolvable Clone()
//            {
//                return new Evolvable
//                       {
//                           Fitness = Fitness,
//                           Chromosome = Chromosome.ToArray()
//                       };
//            }
//        }
//    }
//}