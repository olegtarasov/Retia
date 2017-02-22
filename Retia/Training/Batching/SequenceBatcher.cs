using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Training.Batching
{
	public class SequenceBatcher<TType, TInput> where TType : struct, IEquatable<TType>, IFormattable
    {
		private readonly int _size;
		private readonly Func<TInput, int, TType> _mapper;

		public SequenceBatcher(int size, Func<TInput, int, TType> mapper)
		{
            if (mapper == null)
            {
                throw new ArgumentNullException(nameof(mapper));
            }

            _size = size;
			_mapper = mapper;
		}

		public List<Matrix<TType>> BatchSamples(List<TInput> samples, BatchDimension dimension, IProgressWriter progressWriter = null)
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

		private List<Matrix<TType>> BatchSamples(List<TInput> samples, int batchCount, int batchSize, IProgressWriter progressWriter)
		{
			var result = new List<Matrix<TType>>();
			
			var tracker = new ProgressTracker(batchSize);

			for (int sampleIdx = 0; sampleIdx < batchCount; sampleIdx++)
			{
				if (tracker.ShouldReport(sampleIdx))
				{
				    progressWriter?.SetItemProgress(sampleIdx, batchCount, "Batching");
				}

				var matrix = Matrix<TType>.Build.Dense(_size, batchSize);
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

            progressWriter?.ItemComplete();

			return result;
		}
	}
}