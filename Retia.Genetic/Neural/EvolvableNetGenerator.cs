using Retia.Genetic.Generators;

namespace Retia.Genetic.Neural
{
    public class EvolvableNetGenerator : IPopulationGenerator<EvolvableNet>
    {
        private readonly int _hiddenSize;
        private readonly int _inputSize;
        private readonly int _outputSize;

        public EvolvableNetGenerator(int inputSize, int hiddenSize, int outputSize)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;
            _outputSize = outputSize;
        }

        public EvolvableNet GenerateIndividual()
        {
            return new EvolvableNet(_inputSize, _hiddenSize, _outputSize);
        }
    }
}