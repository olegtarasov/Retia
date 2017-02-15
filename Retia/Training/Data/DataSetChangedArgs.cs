using System;

namespace Retia.Training.Data
{
    public class DataSetChangedArgs<T> : EventArgs where T : struct, IEquatable<T>, IFormattable
    {
        public DataSetChangedArgs(IDataSet<T> oldSet, IDataSet<T> newSet)
        {
            OldSet = oldSet;
            NewSet = newSet;
        }

        public IDataSet<T> OldSet { get; set; }
        public IDataSet<T> NewSet { get; set; }
    }
}