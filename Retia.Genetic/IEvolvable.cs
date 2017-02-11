namespace Retia.Genetic
{
	public interface IEvolvable
	{
		double Fitness { get; set; }
        double[] Chromosome { get; }

		IEvolvable Clone();
	}
}