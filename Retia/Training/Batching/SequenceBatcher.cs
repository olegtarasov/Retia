using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Training.Batching
{
	public class SequenceBatcher : SequenceBatcher<float[]>
	{
		public SequenceBatcher(int size) : base(size, (arr, i) => arr[i])
		{
		}
	}

    public class SequenceBatcher<T>
	{
		private readonly int _size;
		private readonly Func<T, int, float> _mapper;

		public SequenceBatcher(int size, Func<T, int, float> mapper)
		{
			if (mapper == null)
			{
				throw new ArgumentNullException(nameof(mapper));
			}

			_size = size;
			_mapper = mapper;
		}

		public List<Matrix> BatchSamples(List<T> samples, BatchDimension dimension, IProgressWriter progressWriter = null)
		{
			int batchSize, batchCount;
			switch (dimension.Type)
			{
				case BatchDimensionType.BatchSize:
					batchCount = samples.Count / dimension.Dimension;
					batchSize = dimension.Dimension;
					break;
				case BatchDimensionType.BatchCount:
					batchCount = dimension.Dimension;
					batchSize = samples.Count / dimension.Dimension;
					break;
				default:
					throw new ArgumentOutOfRangeException();
			}

			return BatchSamples(samples, batchCount, batchSize, progressWriter);
		}

		private List<Matrix> BatchSamples(List<T> samples, int batchCount, int batchSize, IProgressWriter progressWriter)
		{
			var result = new List<Matrix>();
			
			var tracker = new ProgressTracker(batchSize);

			for (int sampleIdx = 0; sampleIdx < batchCount; sampleIdx++)
			{
				if (tracker.ShouldReport(sampleIdx))
				{
				    progressWriter?.SetProgress(sampleIdx, batchCount, "Batching");
				}

				var matrix = new DenseMatrix(_size, batchSize);
				for (int col = 0; col < batchSize; col++)
				{
					var input = samples[sampleIdx + col * batchCount];
					
					for (int i = 0; i < _size; i++)
					{
						matrix[i, col] = _mapper(input, i);
					}
				}

				result.Add(matrix);
			}

            progressWriter?.Complete();

			return result;
		}
	}
}