using System;
using Retia.Neural;
using Retia.Training.Data;

namespace Retia.Training.Testers
{
    public interface ITester<T> where T : struct, IEquatable<T>, IFormattable
    {
        TestResultBase Test(NeuralNet<T> network, IDataSet<T> testSet);
    }
}