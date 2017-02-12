using System;
using Retia.Integration;

namespace Retia.Training.Data
{
	public interface IDataSet<T> : ICloneable<IDataSet<T>>, IStreamWritable where T : struct, IEquatable<T>, IFormattable
    {
	    event EventHandler DataSetReset;

		Sample<T> GetNextSample();
	    TrainingSequence<T> GetNextSamples(int count);
		void Reset();

		int SampleCount { get; }
		int InputSize { get; }
		int TargetSize { get; }
		int BatchSize { get; }
	}
}