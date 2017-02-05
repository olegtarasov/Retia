namespace Retia.Training.Batching
{
	public class LinearSample
	{
		public LinearSample()
		{
		}

		public LinearSample(float[] input, float[] target)
		{
			Input = input;
			Target = target;
		}

		public float[] Input { get; set; }
		public float[] Target { get; set; } 
	}
}