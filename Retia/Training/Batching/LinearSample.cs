using System;

namespace Retia.Training.Batching
{
	public class LinearSample<T> where T : struct, IEquatable<T>, IFormattable
    {
		public LinearSample()
		{
		}

		public LinearSample(T[] input, T[] target)
		{
			Input = input;
			Target = target;
		}

		public T[] Input { get; set; }
		public T[] Target { get; set; } 
	}
}