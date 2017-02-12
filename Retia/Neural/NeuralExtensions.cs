using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;

namespace Retia.Neural
{
	public static class NeuralExtensions
	{
		public static List<Matrix<T>> Clone<T>(this List<Matrix<T>> list) where T : struct, IEquatable<T>, IFormattable
        {
			return list.Select(x => x.CloneMatrix()).ToList();
		}
	}
}