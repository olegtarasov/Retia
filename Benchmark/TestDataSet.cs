using System;
using System.IO;
using System.Linq;
using Retia.Mathematics;
using Retia.Training.Data;

namespace Benchmark
{
    public class TestDataSet : IDataSet
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly int _batchSize;
        private readonly int _seqLength;

        public TestDataSet(int inputSize, int outputSize, int batchSize, int seqLength)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            _batchSize = batchSize;
            _seqLength = seqLength;
        }

        public IDataSet Clone()
        {
            throw new NotSupportedException();
        }

        public void Save(Stream stream)
        {
            throw new NotSupportedException();
        }

        public event EventHandler DataSetReset;

        public Sample GetNextSample()
        {
            return new Sample(MatrixFactory.RandomMatrix(_inputSize, _batchSize, 1.0f), MatrixFactory.RandomMatrix(_outputSize, _batchSize, 1.0f));
        }

        public TrainingSequence GetNextSamples(int count)
        {
            return new TrainingSequence(
                Enumerable.Range(0, count).Select(x => MatrixFactory.RandomMatrix(_inputSize, _batchSize, 1.0f)).ToList(),
                Enumerable.Range(0, count).Select(x => MatrixFactory.RandomMatrix(_outputSize, _batchSize, 1.0f)).ToList());
        }

        public void Reset()
        {
        }

        public int SampleCount => _seqLength;
        public int InputSize => _inputSize;
        public int TargetSize => _outputSize;
        public int BatchSize => _batchSize;
    }
}