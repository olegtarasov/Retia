using System.Collections.Generic;
using System.Linq;
using Retia.Training.Batching;
using Retia.Training.Data;
using Xunit;
using XunitShould;

namespace Retia.Tests.Training.Batching
{
	public class LinearBatcherTests
	{
		private const int InputSize = 10;
		private const int SetSize = 100;

		[Fact]
		public void CanBatchBySize()
		{
			const int batchCount = 20;
			const int batchSize = 5;

			var samples = GetSamples();
			var batcher = new LinearBatcher();
			var result = batcher.BatchSamples(samples, new BatchDimension(BatchDimensionType.BatchSize, batchSize));

			CheckBatches(result, batchCount, batchSize);
		}

		[Fact]
		public void CanBatchByCount()
		{
			const int batchCount = 25;
			const int batchSize = 4;

			var samples = GetSamples();
			var batcher = new LinearBatcher();
			var result = batcher.BatchSamples(samples, new BatchDimension(BatchDimensionType.BatchCount, batchCount));

			CheckBatches(result, batchCount, batchSize);
		}

		private static void CheckBatches(List<Sample> result, int batchCount, int batchSize)
		{
			result.Count.ShouldEqual(batchCount);
			for (int b = 0; b < batchCount; b++)
			{
				var batch = result[b];
				batch.Input.ColumnCount.ShouldEqual(batchSize);
				batch.Input.RowCount.ShouldEqual(InputSize);
				batch.Target.ColumnCount.ShouldEqual(batchSize);
				batch.Target.RowCount.ShouldEqual(InputSize);

				for (int с = 0; с < batchSize; с++)
				{
					for (int r = 0; r < InputSize; r++)
					{
						batch.Input[r, с].ShouldEqual(с * batchCount + b + r);
					}
				}
			}
		}

		private List<LinearSample> GetSamples()
		{
			return Enumerable.Range(0, SetSize).Select(x => new LinearSample(Enumerable.Range(x, InputSize).Select(d => (float)d).ToArray(), Enumerable.Range(x, InputSize).Select(d => (float)d).ToArray())).ToList();
		}
	}
}