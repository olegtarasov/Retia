using System;
using System.IO;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;

namespace Retia.Training.Data
{
	public class Sample<T> where T : struct, IEquatable<T>, IFormattable
    {
		public Sample()
		{
		}

		public Sample(Matrix<T> input, Matrix<T> target)
		{
			Input = input;
			Target = target;
		}

		public Sample(int inputSize, int targetSize, int batchSize)
		{
			Input = Matrix<T>.Build.Dense(inputSize, batchSize);
			Target = Matrix<T>.Build.Dense(targetSize, batchSize);
		}

		public Matrix<T> Input { get; set; }
		public Matrix<T> Target { get; set; }

		public bool EqualsTo(Sample<T> other)
		{
			if (ReferenceEquals(this, other) || (ReferenceEquals(Input, other.Input) && ReferenceEquals(Target, other.Target)))
			{
				return true;
			}

			return Input.AlmostEqual(other.Input, 1e-5d) && Target.AlmostEqual(other.Target, 1e-5d);
		}

		public void Save(Stream stream)
		{
			MatrixFactory.Save(Input, stream);
			MatrixFactory.Save(Target, stream);
		}

		public static Sample<T> Load(Stream stream)
		{
			return new Sample<T>(MatrixFactory.Load<T>(stream), MatrixFactory.Load<T>(stream));
		}
	}
}