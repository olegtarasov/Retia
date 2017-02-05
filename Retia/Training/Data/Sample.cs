using System.IO;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;

namespace Retia.Training.Data
{
	public class Sample
	{
		public Sample()
		{
		}

		public Sample(Matrix input, Matrix target)
		{
			Input = input;
			Target = target;
		}

		public Sample(int inputSize, int targetSize, int batchSize)
		{
			Input = new DenseMatrix(inputSize, batchSize);
			Target = new DenseMatrix(targetSize, batchSize);
		}

		public Matrix Input { get; set; }
		public Matrix Target { get; set; }

		public bool EqualsTo(Sample other)
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

		public static Sample Load(Stream stream)
		{
			return new Sample(MatrixFactory.Load(stream), MatrixFactory.Load(stream));
		}
	}
}