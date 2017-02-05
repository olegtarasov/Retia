using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;

namespace Retia.Neural
{
	public static class NeuralExtensions
	{
		public static List<Matrix> Clone(this List<Matrix> list)
		{
			return list.Select(x => x.CloneMatrix()).ToList();
		}
	}
}