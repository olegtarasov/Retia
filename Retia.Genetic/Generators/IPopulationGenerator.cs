namespace Retia.Genetic.Generators
{
	public interface IPopulationGenerator<T> where T : IEvolvable
	{
		T GenerateIndividual();
	}
}