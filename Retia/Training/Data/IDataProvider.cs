using System;

namespace Retia.Training.Data
{
	public interface IDataProvider<T> where T : struct, IEquatable<T>, IFormattable
    {
		IDataSet<T> CreateTrainingSet();
		IDataSet<T> CreateTestSet();

        IDataSet<T> TrainingSet { get; }
        IDataSet<T> TestSet { get; }

        int InputSize { get;  }
		int OutputSize { get; }
	    event EventHandler<DataSetChangedArgs<T>> TrainingSetChanged;
	    event EventHandler<DataSetChangedArgs<T>> TestSetChanged;
	}
}