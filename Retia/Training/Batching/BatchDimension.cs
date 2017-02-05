using System;
using System.Collections.Generic;
using System.Linq;

namespace Retia.Training.Batching
{
	public enum BatchDimensionType
	{
		BatchSize,
		BatchCount
	}

	public class BatchDimension
	{
		public BatchDimension(BatchDimensionType type, int dimension)
		{
			Type = type;
			Dimension = dimension;
		}

		public BatchDimensionType Type { get; set; }
		public int Dimension { get; set; }

		// Well, this is a hack
		public IEnumerable<BatchDimensionType> DimensionTypes => Enum.GetValues(typeof(BatchDimensionType)).Cast<BatchDimensionType>();
	}
}