using Retia.Neural;
using Retia.Training.Data;

namespace Retia.Training.Testers
{
    public interface ITester
    {
        TestResultBase Test(NeuralNet network, IDataSet testSet);
    }
}