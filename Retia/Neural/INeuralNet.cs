using Retia.Integration;

namespace Retia.Neural
{
    public interface INeuralNet : IFileWritable
    {
        void Optimize();
        int InputSize { get; }
        int OutputSize { get; }
        int TotalParamCount { get; }
    }
}