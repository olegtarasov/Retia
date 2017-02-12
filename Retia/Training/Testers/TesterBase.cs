using System;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Training.Data;

namespace Retia.Training.Testers
{
    public abstract class TesterBase<T, TResult> : ITester<T>
        where TResult : TestResultBase 
        where T : struct, IEquatable<T>, IFormattable
    {
        protected MathProviderBase<T> MathProvider = MathProvider<T>.Instance;

        public event EventHandler<TResult> TestReport;

        protected abstract TResult TestInternal(NeuralNet<T> network, IDataSet<T> testSet);

        public TestResultBase Test(NeuralNet<T> network, IDataSet<T> testSet)
        {
            network.ResetMemory();
            var result = TestInternal(network, testSet);
            OnTestReport(result);

            return result;
        }

        protected virtual void OnTestReport(TResult e)
        {
            TestReport?.Invoke(this, e);
        }
    }
}