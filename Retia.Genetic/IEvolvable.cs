namespace Retia.Genetic
{
	public interface IEvolvable
	{
		double Fitness { get; set; }
        float[] Chromosome { get; }

		IEvolvable Clone();
	}
}