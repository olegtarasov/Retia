using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Mathematics;
using Retia.Training.Batching;
using Xunit;
using XunitShould;

namespace Retia.Tests.Training.Batching
{
	public class SequenceBatcherTests
	{
		private const int InputSize = 10;
		private const int SetSize = 101;

		[Fact]
		public void CanBatchBySize()
		{
			const int batchCount = 20;
			const int batchSize = 5;

			var samples = GetSamples();
			var batcher = new SequenceBatcher(samples[0].Length);
			var result = batcher.BatchSamples(samples, new BatchDimension(BatchDimensionType.BatchSize, batchSize));

			CheckBatches(result, batchCount, batchSize);
		}

		[Fact]
		public void CanBatchByCount()
		{
			const int batchCount = 25;
			const int batchSize = 4;

			var samples = GetSamples();
			var batcher = new SequenceBatcher(samples[0].Length);
			var result = batcher.BatchSamples(samples, new BatchDimension(BatchDimensionType.BatchCount, batchCount));

			CheckBatches(result, batchCount, batchSize);
		}

		private static void CheckBatches(List<Matrix> result, int batchCount, int batchSize)
		{
			result.Count.ShouldEqual(batchCount);
			
			for (int b = 0; b < batchCount; b++)
			{
				var batch = result[b];
				batch.ColumnCount.ShouldEqual(batchSize);
				batch.RowCount.ShouldEqual(InputSize);
				
				for (int с = 0; с < batchSize; с++)
				{
					for (int r = 0; r < InputSize; r++)
					{
						batch[r, с].ShouldEqual(с * batchCount + b + r);
					}
				}
			}
		}

		private List<float[]> GetSamples()
		{
			return Enumerable.Range(0, SetSize).Select(x => Enumerable.Range(x, InputSize).Select(d => (float)d).ToArray()).ToList();
		}
	}
}