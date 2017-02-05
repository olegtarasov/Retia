using System;
using Retia.Integration;

namespace Retia.Training.Data
{
	public interface IDataSet : ICloneable<IDataSet>, IStreamWritable
	{
	    event EventHandler DataSetReset;

		Sample GetNextSample();
	    TrainingSequence GetNextSamples(int count);
		void Reset();

		int SampleCount { get; }
		int InputSize { get; }
		int TargetSize { get; }
		int BatchSize { get; }
	}
}