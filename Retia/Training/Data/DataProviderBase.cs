using System;
using Retia.Mathematics;

namespace Retia.Training.Data
{
    public abstract class DataProviderBase<T> : IDataProvider<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected static MathProviderBase<T> MathProvider = MathProvider<T>.Instance;

        private IDataSet<T> _trainingSet;
        private IDataSet<T> _testSet;

        public event EventHandler<DataSetChangedArgs<T>> TrainingSetChanged;
        public event EventHandler<DataSetChangedArgs<T>> TestSetChanged;

        public abstract IDataSet<T> CreateTrainingSet();
        public abstract IDataSet<T> CreateTestSet();

        public IDataSet<T> TrainingSet
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
                OnTrainingSetChanged(new DataSetChangedArgs<T>(old, _trainingSet));
            }
        }

        public IDataSet<T> TestSet
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
                OnTestSetChanged(new DataSetChangedArgs<T>(old, _testSet));
            }
        }

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }


        protected virtual void OnTrainingSetChanged(DataSetChangedArgs<T> e)
        {
            TrainingSetChanged?.Invoke(this, e);
        }

        protected virtual void OnTestSetChanged(DataSetChangedArgs<T> e)
        {
            TestSetChanged?.Invoke(this, e);
        }
    }
}