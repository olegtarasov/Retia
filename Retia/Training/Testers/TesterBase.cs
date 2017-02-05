using System;
using Retia.Neural;
using Retia.Training.Data;

namespace Retia.Training.Testers
{
    public abstract class TesterBase<T> : ITester where T : TestResultBase
    {
        public event EventHandler<T> TestReport;

        protected abstract T TestInternal(NeuralNet network, IDataSet testSet);

        public TestResultBase Test(NeuralNet network, IDataSet testSet)
        {
            network.ResetMemory();
            var result = TestInternal(network, testSet);
            OnTestReport(result);

            return result;
        }

        protected virtual void OnTestReport(T e)
        {
            TestReport?.Invoke(this, e);
        }
    }
}