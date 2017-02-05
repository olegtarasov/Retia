using System;

namespace Retia.Training.Data
{
    public abstract class DataProviderBase : IDataProvider
    {
        private IDataSet _trainingSet;
        private IDataSet _testSet;

        public event EventHandler<DataSetChangedArgs> TrainingSetChanged;
        public event EventHandler<DataSetChangedArgs> TestSetChanged;

        public abstract IDataSet CreateTrainingSet();
        public abstract IDataSet CreateTestSet();

        public IDataSet TrainingSet
        {
            get
            {
                if (_trainingSet == null)
                {
                    throw new InvalidOperationException("Training set has not been initialized!");
                }

                return _trainingSet;
            }
            protected set
            {
                var old = _trainingSet;
                _trainingSet = value;
                OnTrainingSetChanged(new DataSetChangedArgs(old, _trainingSet));
            }
        }

        public IDataSet TestSet
        {
            get
            {
                if (_testSet == null)
                {
                    throw new InvalidOperationException("Test set has not been initialized!");
                }

                return _testSet;
            }
            protected set
            {
                var old = _testSet;
                _testSet = value;
                OnTestSetChanged(new DataSetChangedArgs(old, _testSet));
            }
        }

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }


        protected virtual void OnTrainingSetChanged(DataSetChangedArgs e)
        {
            TrainingSetChanged?.Invoke(this, e);
        }

        protected virtual void OnTestSetChanged(DataSetChangedArgs e)
        {
            TestSetChanged?.Invoke(this, e);
        }
    }
}