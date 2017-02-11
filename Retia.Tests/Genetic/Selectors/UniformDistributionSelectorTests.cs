//using System.Linq;
//using Retia.Genetic.Selectors;
//using Retia.Tests.Plumbing;
//using Xunit;
//using XunitShould;

//namespace Retia.Tests.Genetic.Selectors
//{
//    public class UniformDistributionSelectorTests : GeneticTestsBase
//    {
//        protected UniformDistributionSelector<Evolvable> Selector => new UniformDistributionSelector<Evolvable>();

//        [Fact]
//        public void CanSelectSingleDude()
//        {
//            using (new FakeRandom(2))
//            {
//                var pop = GetPopulation();
//                var selected = Selector.GetIndividual(pop, 0);
//                AssertSourceChromosome(selected, 2);
//            }
//        }

//        [Fact]
//        public void CanSelectAllDudes()
//        {
//            var pop = GetPopulation();
//            var dudes = Selector.GetUniqueIndividuals(pop, 10, 0);

//            dudes.ShouldNotBeNull();
//            dudes.Length.ShouldEqual(10);
//            for (int i = 0; i < dudes.Length; i++)
//            {
//                AssertSourceChromosome(dudes[i], i);
//            }
//        }

//        [Fact]
//        public void CanSelectUniqueDudes()
//        {
//            using (new FakeRandom(4, 6))
//            {
//                var pop = GetPopulation();
//                var dudes = Selector.GetUniqueIndividuals(pop, 2, 0);

//                dudes.ShouldNotBeNull();
//                dudes.Length.ShouldEqual(2);

//                AssertSourceChromosome(dudes[0], 4);
//                AssertSourceChromosome(dudes[1], 6);
//            }
//        }

//        [Fact]
//        public void CanSkipNonUniqueDudes()
//        {
//            using (new FakeRandom(4, 6, 6, 6, 8))
//            {
//                var pop = GetPopulation();
//                var dudes = Selector.GetUniqueIndividuals(pop, 3, 0);

//                dudes.ShouldNotBeNull();
//                dudes.Length.ShouldEqual(3);

//                AssertSourceChromosome(dudes[0], 4);
//                AssertSourceChromosome(dudes[1], 6);
//                AssertSourceChromosome(dudes[2], 8);
//            }
//        }
//    }
//}