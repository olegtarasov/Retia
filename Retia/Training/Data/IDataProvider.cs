using System;

namespace Retia.Training.Data
{
	public interface IDataProvider
	{
		IDataSet CreateTrainingSet();
		IDataSet CreateTestSet();

        IDataSet TrainingSet { get; }
        IDataSet TestSet { get; }

        int InputSize { get;  }
		int OutputSize { get; }
	    event EventHandler<DataSetChangedArgs> TrainingSetChanged;
	    event EventHandler<DataSetChangedArgs> TestSetChanged;
	}
}