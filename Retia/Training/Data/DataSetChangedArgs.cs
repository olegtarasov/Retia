using System;

namespace Retia.Training.Data
{
    public class DataSetChangedArgs : EventArgs
    {
        public DataSetChangedArgs(IDataSet oldSet, IDataSet newSet)
        {
            OldSet = oldSet;
            NewSet = newSet;
        }

        public IDataSet OldSet { get; set; }
        public IDataSet NewSet { get; set; }
    }
}